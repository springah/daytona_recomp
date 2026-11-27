#include "dt_prepare.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace dt_prepare {
namespace {

namespace fs = std::filesystem;

// The STFS layout scripts/extract_game.py reads, with its names.
constexpr uint64_t BLOCK_SIZE = 0x1000;
constexpr uint64_t FILE_TABLE_OFFSET = 0xC000;
constexpr size_t FILE_TABLE_ENTRY_SIZE = 0x40;
constexpr uint32_t ROOT_PARENT = 0xFFFF;
constexpr uint32_t END_OF_CHAIN = 0xFFFFFF;

// What the header says about the package, read before anything is unpacked.
constexpr size_t kHeaderSize = 0x364;
constexpr size_t kContentIdOffset = 0x32C;   // 20 bytes, the package's own id
constexpr size_t kContentTypeOffset = 0x344;
constexpr size_t kTitleIdOffset = 0x360;
constexpr uint32_t kArcadeTitle = 0x000D0000;
constexpr uint32_t kDaytonaTitleId = 0x58410B1D;

// game/ holds this file once it is unpacked in full, and only then. Bump
// kFormat when what an unpack leaves in game/ changes, so older ones are
// unpacked again.
constexpr const char* kMarkerName = "daytona-prepared.txt";
constexpr int kFormat = 1;

// Why the step stopped, in words for the player.
struct Failure : std::runtime_error {
    using std::runtime_error::runtime_error;
};

void Progress(double fraction, const std::string& text) {
    std::printf("progress %.3f %s\n", std::clamp(fraction, 0.0, 1.0), text.c_str());
    std::fflush(stdout);
}

// A path as UTF-8 text, for messages. (The build uses -fno-char8_t, so
// u8string() is spelled differently there; this works either way.)
std::string Utf8(const fs::path& path) {
#if defined(_WIN32)
    const std::wstring wide = path.wstring();
    const int size = WideCharToMultiByte(CP_UTF8, 0, wide.data(), int(wide.size()), nullptr, 0,
                                         nullptr, nullptr);
    std::string text(size_t(std::max(size, 0)), '\0');
    WideCharToMultiByte(CP_UTF8, 0, wide.data(), int(wide.size()), text.data(), size, nullptr,
                        nullptr);
    return text;
#else
    return path.string();
#endif
}

uint32_t ReadUint24Le(const uint8_t* data) {
    return data[0] | (data[1] << 8) | (data[2] << 16);
}

uint32_t ReadUint24Be(const uint8_t* data) {
    return (data[0] << 16) | (data[1] << 8) | data[2];
}

uint32_t ReadUint32Be(const uint8_t* data) {
    return (uint32_t(data[0]) << 24) | (data[1] << 16) | (data[2] << 8) | data[3];
}

struct Entry {
    uint32_t index;
    std::string name;
    uint8_t flags;
    uint32_t blocks;
    uint32_t start_block;
    uint32_t parent;
    uint32_t size;

    bool is_directory() const { return (flags & 0x80) != 0; }
};

// Map an STFS logical data block to its physical package block.
uint64_t PhysicalBlock(uint64_t logical_block) {
    const uint64_t group = logical_block / 0xAA;
    const uint64_t level1_groups = group / 0xAA;
    const uint64_t level1_overhead = level1_groups > 0 ? level1_groups + 1 : 0;
    return logical_block + 0x0C + group + (group > 0 ? 1 : 0) + level1_overhead;
}

uint64_t PhysicalOffset(uint64_t logical_block) {
    return PhysicalBlock(logical_block) * BLOCK_SIZE;
}

uint64_t HashEntryOffset(uint64_t logical_block) {
    const uint64_t group = logical_block / 0xAA;
    const uint64_t index = logical_block % 0xAA;
    const uint64_t level1_groups = group / 0xAA;
    const uint64_t level1_overhead = level1_groups > 0 ? level1_groups + 1 : 0;
    const uint64_t table_block = 0x0B + (group * 0xAB) + (group > 0 ? 1 : 0) + level1_overhead;
    return (table_block * BLOCK_SIZE) + (index * 0x18);
}

uint32_t NextBlock(std::ifstream& source, uint64_t logical_block) {
    std::array<uint8_t, 3> raw{};
    source.seekg(static_cast<std::streamoff>(HashEntryOffset(logical_block) + 0x15));
    if (!source.read(reinterpret_cast<char*>(raw.data()), raw.size())) {
        throw Failure("the package ends in the middle of its block table; it is incomplete");
    }
    return ReadUint24Be(raw.data());
}

// A name the file table gives must stay inside the folder it is unpacked to.
bool IsSafeName(const std::string& name) {
    if (name.empty() || name == "." || name == "..") return false;
    return name.find_first_of("/\\:") == std::string::npos;
}

std::vector<Entry> ParseEntries(const fs::path& package) {
    std::vector<Entry> entries;
    std::ifstream f(package, std::ios::binary);
    f.seekg(static_cast<std::streamoff>(FILE_TABLE_OFFSET));
    for (uint32_t index = 0;; ++index) {
        std::array<uint8_t, FILE_TABLE_ENTRY_SIZE> raw{};
        if (!f.read(reinterpret_cast<char*>(raw.data()), raw.size()) ||
            std::all_of(raw.begin(), raw.end(), [](uint8_t b) { return b == 0; })) {
            break;
        }

        const uint8_t name_flags = raw[0x28];
        const size_t name_len = name_flags & 0x3F;
        if (name_len == 0) break;

        std::string name(reinterpret_cast<const char*>(raw.data()), name_len);
        if (!std::all_of(name.begin(), name.end(), [](char c) { return (c & 0x80) == 0; })) {
            throw Failure("the package's file table has a name that is not ASCII");
        }
        if (!IsSafeName(name)) {
            throw Failure("the package's file table has an unusable name: " + name);
        }
        entries.push_back(Entry{index, std::move(name), name_flags, ReadUint24Le(&raw[0x29]),
                                ReadUint24Le(&raw[0x2F]),
                                (uint32_t(raw[0x32]) << 8) | raw[0x33],
                                ReadUint32Be(&raw[0x34])});
    }

    if (std::none_of(entries.begin(), entries.end(),
                     [](const Entry& e) { return e.name == "default.xex"; })) {
        throw Failure("could not find default.xex in the STFS file table");
    }
    return entries;
}

fs::path EntryPath(const Entry& entry, const std::vector<Entry>& entries) {
    std::vector<const std::string*> parts{&entry.name};
    std::set<uint32_t> seen{entry.index};
    for (uint32_t parent = entry.parent; parent != ROOT_PARENT;) {
        if (parent >= entries.size() || seen.count(parent) != 0) {
            throw Failure("invalid parent chain for " + entry.name);
        }
        parts.push_back(&entries[parent].name);
        seen.insert(parent);
        parent = entries[parent].parent;
    }
    fs::path path;
    for (auto it = parts.rbegin(); it != parts.rend(); ++it) path /= **it;   // ASCII, checked above
    return path;
}

// Copies one file out of its block chain. `on_bytes` is told every block.
template <typename OnBytes>
void ExtractFile(const fs::path& package, const Entry& entry, const fs::path& destination,
                 OnBytes on_bytes) {
    fs::create_directories(destination.parent_path());
    uint64_t remaining = entry.size;
    const uint64_t blocks_to_copy =
        std::max<uint64_t>(entry.blocks, (uint64_t(entry.size) + BLOCK_SIZE - 1) / BLOCK_SIZE);
    uint64_t logical_block = entry.start_block;

    std::ifstream source(package, std::ios::binary);
    std::ofstream output(destination, std::ios::binary | std::ios::trunc);
    if (!output) throw Failure("could not write " + Utf8(destination));
    std::array<char, BLOCK_SIZE> chunk{};
    for (uint64_t block_index = 0; block_index < blocks_to_copy; ++block_index) {
        if (remaining == 0) break;
        if (logical_block == END_OF_CHAIN) {
            throw Failure("unexpected end of block chain while extracting " + entry.name);
        }

        source.seekg(static_cast<std::streamoff>(PhysicalOffset(logical_block)));
        const size_t want = static_cast<size_t>(std::min<uint64_t>(BLOCK_SIZE, remaining));
        source.read(chunk.data(), static_cast<std::streamsize>(want));
        const size_t got = static_cast<size_t>(source.gcount());
        if (got == 0) throw Failure("unexpected EOF while extracting " + entry.name);
        source.clear();
        output.write(chunk.data(), static_cast<std::streamsize>(got));
        remaining -= got;
        on_bytes(got);

        if (block_index + 1 < blocks_to_copy) {
            logical_block = NextBlock(source, logical_block);
        }
    }
    if (!output.flush()) throw Failure("could not write " + Utf8(destination) + " (is the disk full?)");
}

// Which copy game/ was unpacked from: its size and the id in its header.
struct Identity {
    uint64_t size = 0;
    std::string content_id;   // hex

    bool operator==(const Identity&) const = default;
};

Identity ReadIdentity(const fs::path& package) {
    std::error_code ec;
    if (fs::is_directory(package, ec)) {
        throw Failure(Utf8(package) + " is a folder. Pick the package file itself.");
    }
    if (!fs::is_regular_file(package, ec)) {
        throw Failure("Your package was not found: " + Utf8(package));
    }
    std::array<uint8_t, kHeaderSize> header{};
    std::ifstream f(package, std::ios::binary);
    if (!f.read(reinterpret_cast<char*>(header.data()), header.size()) ||
        header[0] != 'L' || header[1] != 'I' || header[2] != 'V' || header[3] != 'E') {
        throw Failure(Utf8(package.filename()) +
                      " is not an Xbox 360 LIVE/STFS package. Pick the package itself (a file "
                      "with no extension and a long hexadecimal name), not a file from inside it.");
    }
    const uint32_t title_id = ReadUint32Be(&header[kTitleIdOffset]);
    if (title_id != kDaytonaTitleId) {
        char text[160];
        std::snprintf(text, sizeof(text),
                      "This package is another game's (title ID %08X). Daytona USA's is %08X.",
                      title_id, kDaytonaTitleId);
        throw Failure(text);
    }
    const uint32_t content_type = ReadUint32Be(&header[kContentTypeOffset]);
    if (content_type != kArcadeTitle) {
        char text[200];
        std::snprintf(text, sizeof(text),
                      "This Daytona USA package is not the game itself (content type %08X): pick "
                      "the Arcade title package, not an update or add-on.",
                      content_type);
        throw Failure(text);
    }

    Identity identity;
    identity.size = fs::file_size(package);
    for (size_t i = 0; i < 20; ++i) {
        char hex[3];
        std::snprintf(hex, sizeof(hex), "%02x", header[kContentIdOffset + i]);
        identity.content_id += hex;
    }
    return identity;
}

// The identity game/ was unpacked from, when it is unpacked in full by this
// format; nothing otherwise.
std::optional<Identity> ReadMarker(const fs::path& game) {
    std::error_code ec;
    if (!fs::is_regular_file(game / "default.xex", ec)) return std::nullopt;
    std::ifstream in(game / kMarkerName);
    std::string line;
    std::optional<int> format;
    Identity identity;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const size_t eq = line.find(" = ");
        if (line.empty() || line[0] == '#' || eq == std::string::npos) continue;
        const std::string key = line.substr(0, eq);
        const std::string value = line.substr(eq + 3);
        try {
            if (key == "format") format = std::stoi(value);
            if (key == "package_size") identity.size = std::stoull(value);
        } catch (const std::exception&) {
            return std::nullopt;
        }
        if (key == "content_id") identity.content_id = value;
    }
    if (format != kFormat || identity.content_id.empty()) return std::nullopt;
    return identity;
}

void WriteMarker(const fs::path& game, const Identity& identity) {
    std::ofstream out(game / kMarkerName, std::ios::binary | std::ios::trunc);
    out << "# Written by daytona --prepare once game/ was unpacked in full.\n"
        << "format = " << kFormat << "\n"
        << "package_size = " << identity.size << "\n"
        << "content_id = " << identity.content_id << "\n";
    if (!out.flush()) throw Failure("could not write " + Utf8(game / kMarkerName));
}

// Held for as long as the step runs. The OS lets go of it when the process
// ends however it ends, so a run the launcher cancelled (it kills the step)
// never leaves the next one locked out; the file itself staying is harmless.
class RunLock {
public:
    explicit RunLock(const fs::path& path) {
#if defined(_WIN32)
        handle_ = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_ALWAYS,
                              FILE_ATTRIBUTE_NORMAL, nullptr);
        held_ = handle_ != INVALID_HANDLE_VALUE;
        busy_ = !held_ && GetLastError() == ERROR_SHARING_VIOLATION;
#else
        fd_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
        if (fd_ >= 0 && ::flock(fd_, LOCK_EX | LOCK_NB) == 0) {
            held_ = true;
        } else {
            busy_ = fd_ >= 0 && errno == EWOULDBLOCK;
        }
#endif
    }
    ~RunLock() {
#if defined(_WIN32)
        if (held_) CloseHandle(handle_);
#else
        if (fd_ >= 0) ::close(fd_);
#endif
    }
    RunLock(const RunLock&) = delete;
    RunLock& operator=(const RunLock&) = delete;

    bool held() const { return held_; }
    bool busy() const { return busy_; }

private:
#if defined(_WIN32)
    HANDLE handle_ = INVALID_HANDLE_VALUE;
#else
    int fd_ = -1;
#endif
    bool held_ = false;
    bool busy_ = false;
};

// daytona.exe is a windowed program, so a terminal that starts it by hand
// gives it no console. Use that terminal's, so the progress lines show; a
// launcher that hands it pipes keeps them.
void UseParentConsole() {
#if defined(_WIN32)
    const HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
    if ((out == nullptr || out == INVALID_HANDLE_VALUE) && AttachConsole(ATTACH_PARENT_PROCESS)) {
        std::freopen("CONOUT$", "w", stdout);
        std::freopen("CONOUT$", "w", stderr);
    }
#endif
}

// Every file and folder of the package into `out`, in file table order, as
// extract_game.py writes them. Returns how many files it wrote.
size_t UnpackInto(const fs::path& package, const fs::path& out) {
    const std::vector<Entry> entries = ParseEntries(package);
    uint64_t total = 0;
    for (const Entry& entry : entries) {
        if (!entry.is_directory()) total += entry.size;
    }

    uint64_t done = 0;
    size_t extracted = 0;
    bool root_xex = false;
    fs::create_directories(out);
    for (const Entry& entry : entries) {
        const fs::path relative = EntryPath(entry, entries);
        const fs::path destination = out / relative;
        if (entry.is_directory()) {
            fs::create_directories(destination);
            continue;
        }
        const std::string label = "Unpacking " + relative.generic_string();
        Progress(total ? 0.97 * double(done) / double(total) : 0.0, label);
        uint64_t next_report = done + (16u << 20);
        ExtractFile(package, entry, destination, [&](size_t bytes) {
            done += bytes;
            if (done >= next_report) {
                Progress(0.97 * double(done) / double(total), label);
                next_report = done + (16u << 20);
            }
        });
        ++extracted;
        if (entry.name == "default.xex" && entry.parent == ROOT_PARENT) root_xex = true;
    }
    if (!root_xex) throw Failure("extracted package did not contain root default.xex");
    return extracted;
}

int Unpack(const fs::path& package, const fs::path& data_root) {
    const fs::path game = data_root / "game";
    const fs::path partial = data_root / "game.partial";

    std::error_code ec;
    fs::create_directories(data_root, ec);
    RunLock lock(data_root / "daytona-prepare.lock");
    if (!lock.held()) {
        throw Failure(lock.busy() ? "Daytona USA is already being set up. Wait for that to finish, "
                                    "then try again."
                                  : "Could not write to " + Utf8(data_root) + ".");
    }
    // What a run that was cut short left behind. Only a run holding the lock
    // gets here, so no other one is still writing it.
    fs::remove_all(partial);

    Progress(0.0, "Checking your package");
    const std::optional<Identity> unpacked = ReadMarker(game);
    std::error_code exists_ec;
    const bool have_package = !package.empty() && fs::exists(package, exists_ec);
    if (unpacked && !have_package) {
        Progress(1.0, "Already set up");
        return 0;
    }
    if (!have_package) {
        throw Failure(package.empty()
                          ? "The game is not unpacked yet. Pick your Daytona USA package "
                            "(--game=<package>)."
                          : "Your package was not found: " + Utf8(package));
    }
    const Identity identity = ReadIdentity(package);
    if (unpacked && *unpacked == identity) {
        Progress(1.0, "Already set up");
        return 0;
    }

    size_t extracted = 0;
    try {
        extracted = UnpackInto(package, partial);
        Progress(0.98, "Finishing");
        WriteMarker(partial, identity);
    } catch (...) {
        std::error_code ignored;
        fs::remove_all(partial, ignored);   // a failed unpack leaves nothing behind
        throw;
    }
    fs::remove_all(game);
    fs::rename(partial, game);
    std::printf("Extracted %zu files to %s\n", extracted, Utf8(game).c_str());
    Progress(1.0, "Ready to play");
    return 0;
}

}  // namespace

int Run(const std::filesystem::path& package, const std::filesystem::path& data_root) {
    UseParentConsole();
    try {
        if (data_root.empty()) {
            throw Failure("--prepare needs --data_root=<folder>: the game is unpacked into its "
                          "game/ folder, and --data_root=<folder> plays it from there.");
        }
        return Unpack(package, data_root);
    } catch (const Failure& failure) {
        std::fflush(stdout);
        std::fprintf(stderr, "error: %s\n", failure.what());
    } catch (const std::exception& error) {
        // Filesystem errors: a full disk, a folder that can't be written.
        std::fflush(stdout);
        std::fprintf(stderr, "error: %s\n", error.what());
    }
    std::fflush(stderr);
    return 1;
}

}  // namespace dt_prepare

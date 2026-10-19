// ps2kit runtime — the disc: a .cue sheet (with its .bin tracks), a plain
// .iso, or a single raw .bin; its ISO 9660 file system and SYSTEM.CNF.
// The same reading as ps2kit/disc.py: a MODE2/2352 sector keeps its 2048
// user bytes (Form 1) after 24 bytes of sync, header and subheader, a
// MODE1/2352 one after 16; track LBAs count the pregaps.
#include "os.h"

#include <cctype>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>
#ifndef _WIN32
#define _fseeki64 fseeko
#define _ftelli64 ftello
#endif

namespace {

struct Track {
    std::string path, mode;
    int number = 0;
    uint64_t file_offset = 0;            // byte offset of INDEX 01 in the file
    uint32_t pregap = 0, gap = 0;        // frames: INDEX 00 to 01, and a PREGAP not in the file
    uint32_t sector_size = 2352, user_offset = 24;
    uint32_t frames = 0, lba = 0;
    FILE* f = nullptr;
};

std::mutex g_disc;
std::vector<Track> g_tracks;

uint32_t msf(const std::string& s) {
    unsigned m = 0, sec = 0, fr = 0;
    std::sscanf(s.c_str(), "%u:%u:%u", &m, &sec, &fr);
    return (m * 60 + sec) * 75 + fr;
}

uint64_t file_size(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return 0;
    _fseeki64(f, 0, SEEK_END);
    uint64_t n = (uint64_t)_ftelli64(f);
    std::fclose(f);
    return n;
}

void set_mode(Track& t, const std::string& mode) {
    t.mode = mode;
    t.sector_size = (mode == "MODE1/2048" || mode == "MODE2/2048") ? 2048 : 2352;
    t.user_offset = mode.rfind("MODE2", 0) == 0 ? 24 : 16;
}

bool parse_cue(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f) return false;
    std::string base = path.substr(0, path.find_last_of("\\/") + 1);
    std::string cur_file;
    std::vector<int> idx00;
    char line[1024];
    while (std::fgets(line, sizeof line, f)) {
        std::string s = line;
        while (!s.empty() && std::isspace((unsigned char)s.back())) s.pop_back();
        size_t a = s.find_first_not_of(" \t");
        if (a == std::string::npos) continue;
        s = s.substr(a);
        if (s.rfind("FILE ", 0) == 0) {
            size_t q1 = s.find('"'), q2 = s.rfind('"');
            std::string name = q1 != std::string::npos && q2 > q1 ? s.substr(q1 + 1, q2 - q1 - 1)
                                                                   : s.substr(5, s.find(' ', 5) - 5);
            cur_file = base + name;
        } else if (s.rfind("TRACK ", 0) == 0) {
            Track t;
            char mode[32] = {};
            std::sscanf(s.c_str(), "TRACK %d %31s", &t.number, mode);
            t.path = cur_file;
            set_mode(t, mode);
            g_tracks.push_back(t);
            idx00.push_back(-1);
        } else if (s.rfind("PREGAP ", 0) == 0 && !g_tracks.empty()) {
            g_tracks.back().gap = msf(s.substr(7));
        } else if (s.rfind("INDEX ", 0) == 0 && !g_tracks.empty()) {
            int n = 0;
            char t[32] = {};
            std::sscanf(s.c_str(), "INDEX %d %31s", &n, t);
            Track& tr = g_tracks.back();
            if (n == 0) idx00.back() = (int)msf(t);
            else if (n == 1) {
                uint32_t at = msf(t);
                tr.file_offset = (uint64_t)at * tr.sector_size;
                tr.pregap = idx00.back() >= 0 ? at - (uint32_t)idx00.back() : 0;
            }
        }
    }
    std::fclose(f);
    if (g_tracks.empty()) return false;
    for (size_t k = 0; k < g_tracks.size(); ++k) {
        Track& t = g_tracks[k];
        uint64_t end = file_size(t.path);
        if (k + 1 < g_tracks.size() && g_tracks[k + 1].path == t.path)
            end = g_tracks[k + 1].file_offset - (uint64_t)g_tracks[k + 1].pregap * g_tracks[k + 1].sector_size;
        t.frames = (uint32_t)((end - t.file_offset) / t.sector_size);
    }
    uint32_t lba = 0;
    for (Track& t : g_tracks) {
        lba += t.gap + t.pregap;
        t.lba = lba;
        lba += t.frames;
    }
    return true;
}

bool open_single(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    uint8_t head[16] = {};
    size_t n = std::fread(head, 1, 16, f);
    std::fclose(f);
    static const uint8_t sync[12] = {0, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0};
    Track t;
    t.number = 1;
    t.path = path;
    set_mode(t, n == 16 && std::memcmp(head, sync, 12) == 0 ? (head[15] == 2 ? "MODE2/2352" : "MODE1/2352")
                                                              : "MODE1/2048");
    t.frames = (uint32_t)(file_size(path) / t.sector_size);
    g_tracks.push_back(t);
    return true;
}

Track& track_at(uint32_t lba) {
    Track* found = &g_tracks[0];
    for (Track& t : g_tracks)
        if (t.lba - t.pregap - t.gap <= lba) found = &t;
    return *found;
}

bool read_user(uint32_t lba, uint8_t* out) {
    Track& t = track_at(lba);
    if (lba < t.lba - t.pregap) { std::memset(out, 0, 2048); return true; }
    if (!t.f && !(t.f = std::fopen(t.path.c_str(), "rb"))) return false;
    uint64_t at = t.file_offset + (uint64_t)(lba - t.lba) * t.sector_size;
    if (t.sector_size == 2352) at += t.user_offset;
    if (_fseeki64(t.f, (int64_t)at, SEEK_SET) != 0) return false;
    return std::fread(out, 1, 2048, t.f) == 2048;
}

std::string upper(std::string s) {
    for (char& ch : s) ch = (char)std::toupper((unsigned char)ch);
    return s;
}

}  // namespace

bool disc_open(const char* path) {
    std::lock_guard<std::mutex> lk(g_disc);
    std::string p = path;
    std::string ext = upper(p.substr(p.find_last_of('.') + 1));
    bool ok = ext == "CUE" ? parse_cue(p) : open_single(p);
    if (!ok) return false;
    uint8_t pvd[2048];
    return read_user(16, pvd) && std::memcmp(pvd + 1, "CD001", 5) == 0;
}

bool disc_read(uint32_t lsn, uint32_t count, uint8_t* out) {
    std::lock_guard<std::mutex> lk(g_disc);
    for (uint32_t k = 0; k < count; ++k)
        if (!read_user(lsn + k, out + (size_t)k * 2048)) return false;
    return true;
}

// A path as the game writes it: \DIR\NAME.EXT;1 (or with / separators).
// Directory records' names are compared without case and without ";1".
bool disc_find(const std::string& path, DiscFile& out) {
    std::vector<std::string> parts;
    std::string cur;
    for (char ch : path) {
        if (ch == '\\' || ch == '/') { if (!cur.empty()) parts.push_back(cur); cur.clear(); }
        else cur += ch;
    }
    if (!cur.empty()) parts.push_back(cur);
    for (auto& p : parts) p = upper(p.substr(0, p.find(';')));
    std::lock_guard<std::mutex> lk(g_disc);
    uint8_t sec[2048];
    if (!read_user(16, sec)) return false;
    uint32_t lba, size;
    std::memcpy(&lba, sec + 156 + 2, 4);
    std::memcpy(&size, sec + 156 + 10, 4);
    for (size_t k = 0; k < parts.size(); ++k) {
        bool found = false;
        for (uint32_t s = 0; s < (size + 2047) / 2048 && !found; ++s) {
            if (!read_user(lba + s, sec)) return false;
            for (uint32_t pos = 0; pos < 2048 && sec[pos];) {
                const uint8_t* r = sec + pos;
                uint8_t nlen = r[32];
                std::string name((const char*)r + 33, nlen);
                if (upper(name.substr(0, name.find(';'))) == parts[k]) {
                    std::memcpy(&lba, r + 2, 4);
                    std::memcpy(&size, r + 10, 4);
                    if (k + 1 == parts.size()) {
                        out.lsn = lba;
                        out.size = size;
                        std::memset(out.name, 0, sizeof out.name);
                        std::memcpy(out.name, name.data(), std::min<size_t>(name.size(), sizeof out.name));
                        std::memcpy(out.date, r + 18, 7);
                        out.is_dir = r[25] & 2;
                        return true;
                    }
                    found = true;
                    break;
                }
                pos += r[0];
            }
        }
        if (!found) return false;
    }
    return false;
}

bool disc_read_file(const std::string& path, std::vector<uint8_t>& out) {
    DiscFile f;
    if (!disc_find(path, f) || f.is_dir) return false;
    std::vector<uint8_t> buf((size_t)((f.size + 2047) / 2048) * 2048);
    if (!disc_read(f.lsn, (uint32_t)(buf.size() / 2048), buf.data())) return false;
    buf.resize(f.size);
    out.swap(buf);
    return true;
}

// SYSTEM.CNF's BOOT2 line: the executable the boot ROM starts.
std::string disc_boot_path() {
    std::vector<uint8_t> cnf;
    if (!disc_read_file("SYSTEM.CNF", cnf)) return "";
    std::string s(cnf.begin(), cnf.end());
    size_t a = s.find("BOOT2");
    if (a == std::string::npos) return "";
    a = s.find('=', a);
    if (a == std::string::npos) return "";
    size_t b = s.find_first_of("\r\n", a);
    std::string v = s.substr(a + 1, b == std::string::npos ? std::string::npos : b - a - 1);
    v.erase(0, v.find_first_not_of(" \t"));
    while (!v.empty() && std::isspace((unsigned char)v.back())) v.pop_back();
    return v;
}

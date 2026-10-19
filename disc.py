"""PS2 disc images: .cue/.bin sets (CD) and plain .iso (CD or DVD); SYSTEM.CNF, ISO 9660.

    python -m ps2kit.disc GAME.cue --info
    python -m ps2kit.disc GAME.cue --list [--forms]
    python -m ps2kit.disc GAME.cue --extract out/        # files as the drive reads them
    python -m ps2kit.disc GAME.cue --audio out/          # CD-DA tracks as WAV

Layers, from the outside in (the CD part is the same as saturnkit's reader,
which it was adapted from):

CUE   One FILE per track (Redump) or one FILE for all. A PS2 CD's track 1 is
      data, MODE2/2352 (CD-ROM XA); later tracks, when present, are CD-DA
      (AUDIO, 2352-byte frames of 16-bit stereo 44.1 kHz, little-endian).
      INDEX 00 is the pregap, INDEX 01 the start of the track. Disc time
      (MSF, 75 frames a second) starts at 00:02:00 = LBA 0. A PS2 DVD is a
      plain 2048-byte .iso.

Data  A MODE2/2352 sector is 12 sync bytes, 4 header bytes (M S F mode), an
      8-byte subheader (file number, channel, submode, coding info, twice),
      then 2048 user bytes (Form 1, submode bit 5 clear) or 2324 (Form 2,
      bit 5 set). The PS2's sceCdRead reads Form 1 user data; Form 2 sectors
      only matter to streams read in their raw form (XA audio, some movie
      streamers), so `forms()` counts them per file.

ISO   ISO 9660 from sector 16 (PVD). SYSTEM.CNF in the root names the boot
      executable (BOOT2 = cdrom0:\\SLES_123.45;1), the disc's version (VER)
      and its video mode (VMODE = NTSC or PAL). Sectors 0-15 hold the
      PlayStation 2 logo the boot ROM checks; their contents are not used
      above this layer.
"""
import argparse
import os
import re
import struct

SECTOR_RAW = 2352
SECTOR = 2048
FORM2 = 2324


# ---------------------------------------------------------------- CUE / tracks

class Track:
    def __init__(self, number, mode, path, file_offset, pregap):
        self.number, self.mode, self.path = number, mode, path
        self.file_offset = file_offset   # byte offset of INDEX 01 in path
        self.pregap = pregap             # frames between INDEX 00 and 01
        self.gap = 0                     # frames of a PREGAP not in the file, before those
        self.sector_size = 2048 if mode in ("MODE1/2048", "MODE2/2048") else SECTOR_RAW
        self.user_offset = 24 if mode.startswith("MODE2") else 16
        self.frames = 0                  # filled in by parse_cue
        self.lba = 0                     # disc LBA of INDEX 01

    @property
    def is_audio(self):
        return self.mode == "AUDIO"

    def __repr__(self):
        return "<Track %02d %s lba=%d frames=%d>" % (self.number, self.mode,
                                                     self.lba, self.frames)


def _msf(s):
    m, sec, f = (int(x) for x in s.split(":"))
    return (m * 60 + sec) * 75 + f


def parse_cue(path):
    """Tracks of a .cue sheet, with their disc LBAs (pregaps counted)."""
    base = os.path.dirname(os.path.abspath(path))
    tracks, cur_file, cur = [], None, None
    idx00 = {}
    for line in open(path, encoding="latin-1"):
        tok = line.strip()
        m = re.match(r'FILE\s+"(.*)"\s+(\S+)', tok) or re.match(r"FILE\s+(\S+)\s+(\S+)", tok)
        if m:
            cur_file = os.path.join(base, m.group(1))
            continue
        m = re.match(r"TRACK\s+(\d+)\s+(\S+)", tok)
        if m:
            cur = Track(int(m.group(1)), m.group(2), cur_file, None, 0)
            tracks.append(cur)
            continue
        m = re.match(r"PREGAP\s+(\S+)", tok)
        if m and cur:
            cur.gap = _msf(m.group(1))
            continue
        m = re.match(r"INDEX\s+(\d+)\s+(\S+)", tok)
        if m and cur:
            n, t = int(m.group(1)), _msf(m.group(2))
            if n == 0:
                idx00[cur.number] = t
            elif n == 1:
                cur.file_offset = t * cur.sector_size
                cur.pregap = t - idx00.get(cur.number, t)
    by_file = {}
    for t in tracks:
        by_file.setdefault(t.path, []).append(t)
    for f, ts in by_file.items():
        size = os.path.getsize(f)
        for i, t in enumerate(ts):
            end = ts[i + 1].file_offset - ts[i + 1].pregap * ts[i + 1].sector_size \
                if i + 1 < len(ts) else size
            t.frames = (end - t.file_offset) // t.sector_size
    lba = 0
    for t in tracks:
        lba += t.gap + t.pregap
        t.lba = lba
        lba += t.frames
    return tracks


class Disc:
    """A PS2 disc: tracks, the data track's user sectors, ISO 9660, SYSTEM.CNF."""

    def __init__(self, path):
        if path.lower().endswith(".cue"):
            self.tracks = parse_cue(path)
        else:                                   # plain .iso or single .bin
            size = os.path.getsize(path)
            with open(path, "rb") as f:
                head = f.read(16)
            if head[:12] == b"\x00" + b"\xff" * 10 + b"\x00":
                mode = "MODE2/2352" if head[15] == 2 else "MODE1/2352"
            else:
                mode = "MODE1/2048"
            t = Track(1, mode, path, 0, 0)
            t.frames = size // t.sector_size
            self.tracks = [t]
        self.data = self.tracks[0]
        self._fh = {}
        self.iso = ISO9660(self)
        try:
            self.cnf = SystemCnf(self.read_file(self.iso.find("SYSTEM.CNF")))
        except KeyError:
            self.cnf = None

    @property
    def is_dvd(self):
        return self.data.sector_size == SECTOR and self.iso.volume_size > 360000

    def track_at(self, lba):
        """The track holding disc LBA `lba` (its pregap counted as its own)."""
        found = self.tracks[0]
        for t in self.tracks:
            if t.lba - t.pregap - t.gap <= lba:
                found = t
        return found

    def raw_sector(self, lba):
        """(stored bytes, track) of the sector at disc LBA `lba`: 2352 bytes, or 2048 for a plain .iso."""
        t = self.track_at(lba)
        if lba < t.lba - t.pregap:              # a PREGAP that is not in the file
            return bytes(t.sector_size), t
        fh = self._fh.get(t.path)
        if fh is None:
            fh = self._fh[t.path] = open(t.path, "rb")
        fh.seek(t.file_offset + (lba - t.lba) * t.sector_size)
        return fh.read(t.sector_size), t

    def read_sectors(self, lba, count=1):
        """User bytes (2048 a sector, Form 1) of `count` data sectors from disc LBA `lba`."""
        out = bytearray()
        for i in range(count):
            raw, t = self.raw_sector(lba + i)
            out += raw[t.user_offset:t.user_offset + SECTOR] if t.sector_size == SECTOR_RAW else raw
        return bytes(out)

    def read(self, lba, size):
        return self.read_sectors(lba, (size + SECTOR - 1) // SECTOR)[:size]

    def read_file(self, rec):
        return self.read(rec.lba, rec.size)

    def forms(self, rec):
        """(form 1, form 2) sector counts of a file; (n, 0) when the image keeps no subheaders."""
        n = (rec.size + SECTOR - 1) // SECTOR
        t = self.track_at(rec.lba)
        if t.sector_size != SECTOR_RAW or not t.mode.startswith("MODE2"):
            return n, 0
        f2 = 0
        for i in range(n):
            raw, _ = self.raw_sector(rec.lba + i)
            f2 += bool(raw[18] & 0x20)
        return n - f2, f2

    def is_cdda(self, rec):
        return not rec.is_dir and self.track_at(rec.lba).is_audio


# ---------------------------------------------------------------- SYSTEM.CNF

class SystemCnf:
    """BOOT2 (or BOOT, on a PS1 disc), VER, VMODE and anything else, as key -> value."""

    def __init__(self, data):
        self.raw = data
        self.fields = {}
        for line in data.decode("latin-1").splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                self.fields[k.strip().upper()] = v.strip()
        self.boot = self.fields.get("BOOT2") or self.fields.get("BOOT")
        self.is_ps2 = "BOOT2" in self.fields
        self.version = self.fields.get("VER")
        self.vmode = self.fields.get("VMODE")

    @property
    def boot_path(self):
        """The boot executable as a disc path: cdrom0:\\SLES_123.45;1 -> /SLES_123.45."""
        if not self.boot:
            return None
        p = self.boot.split(":", 1)[-1].replace("\\", "/").split(";")[0]
        return "/" + p.lstrip("/")


# ---------------------------------------------------------------- ISO 9660

class Record:
    def __init__(self, name, lba, size, is_dir, path, xa_attr=None, xa_file=0):
        self.name, self.lba, self.size, self.is_dir, self.path = name, lba, size, is_dir, path
        self.xa_attr, self.xa_file = xa_attr, xa_file  # CD-ROM XA attributes and file number

    def __repr__(self):
        return "<%s %s lba=%d size=%d>" % ("dir" if self.is_dir else "file", self.path, self.lba, self.size)


class ISO9660:
    def __init__(self, disc):
        self.disc = disc
        pvd = disc.read_sectors(16)
        if pvd[1:6] != b"CD001":
            raise ValueError("no ISO 9660 primary volume descriptor at sector 16")
        s = lambda a, b: pvd[a:b].decode("latin-1").rstrip()
        self.system_id = s(8, 40)
        self.volume_id = s(40, 72)
        self.volume_size = struct.unpack("<I", pvd[80:84])[0]
        self.volume_set = s(190, 318)
        self.publisher = s(318, 446)
        self.preparer = s(446, 574)
        self.application = s(574, 702)
        self.created = s(813, 830)
        root = pvd[156:190]
        self.root = Record("", struct.unpack("<I", root[2:6])[0], struct.unpack("<I", root[10:14])[0], True, "/")

    def listdir(self, rec):
        data = self.disc.read(rec.lba, rec.size)
        out, pos = [], 0
        while pos < len(data):
            ln = data[pos]
            if ln == 0:                       # records never cross a sector
                pos = (pos // SECTOR + 1) * SECTOR
                continue
            r = data[pos:pos + ln]
            lba, size = struct.unpack("<I", r[2:6])[0], struct.unpack("<I", r[10:14])[0]
            flags, nlen = r[25], r[32]
            name = r[33:33 + nlen]
            su = r[33 + nlen + (1 - nlen % 2):]      # system use, after the name's padding
            xa_attr, xa_file = (struct.unpack(">H", su[4:6])[0], su[8]) if su[6:8] == b"XA" else (None, 0)
            pos += ln
            if name in (b"\x00", b"\x01"):
                continue
            name = name.decode("latin-1").split(";")[0]
            out.append(Record(name, lba, size, bool(flags & 2),
                              rec.path.rstrip("/") + "/" + name, xa_attr, xa_file))
        return out

    def walk(self, rec=None):
        for r in self.listdir(rec or self.root):
            yield r
            if r.is_dir:
                yield from self.walk(r)

    def find(self, path):
        want = path.strip("/").upper()
        for r in self.walk():
            if r.path.strip("/").upper() == want:
                return r
        raise KeyError(path)


# ---------------------------------------------------------------- CD-DA

def audio_frames(track):
    """Raw 16-bit LE stereo PCM of an audio track, INDEX 01 onward."""
    with open(track.path, "rb") as f:
        f.seek(track.file_offset)
        return f.read(track.frames * SECTOR_RAW)


def write_wav(path, pcm, rate=44100, channels=2):
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE")
        f.write(b"fmt " + struct.pack("<IHHIIHH", 16, 1, channels, rate,
                                      rate * channels * 2, channels * 2, 16))
        f.write(b"data" + struct.pack("<I", len(pcm)) + pcm)


# ---------------------------------------------------------------- CLI

def _time(frames):
    s = frames / 75
    return "%d:%05.2f" % (s // 60, s % 60)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("image")
    ap.add_argument("--info", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--forms", action="store_true", help="with --list: Form 1 / Form 2 sectors per file")
    ap.add_argument("--extract", metavar="DIR")
    ap.add_argument("--audio", metavar="DIR")
    a = ap.parse_args(argv)
    d = Disc(a.image)
    if a.info or not (a.list or a.extract or a.audio):
        iso = d.iso
        print("medium     %s, %d sectors" % ("DVD" if d.is_dvd else "CD", iso.volume_size))
        print("volume     %r  system %r  set %r" % (iso.volume_id, iso.system_id, iso.volume_set))
        print("publisher  %r  preparer %r" % (iso.publisher, iso.preparer))
        print("app        %r  created %s" % (iso.application, iso.created))
        if d.cnf:
            print("SYSTEM.CNF %s" % "  ".join("%s=%s" % kv for kv in d.cnf.fields.items()))
        print("tracks:")
        for t in d.tracks:
            print("  %02d %-10s lba %6d  %6d frames  %s%s" % (
                t.number, t.mode, t.lba, t.frames, _time(t.frames),
                "  pregap %d" % (t.gap + t.pregap) if t.gap + t.pregap else ""))
    if a.list:
        for r in d.iso.walk():
            extra = ""
            if d.is_cdda(r):
                extra = "  CD-DA, track %02d" % d.track_at(r.lba).number
            elif a.forms and not r.is_dir:
                f1, f2 = d.forms(r)
                extra = "  form2 %d/%d" % (f2, f1 + f2) if f2 else ""
            print("%-40s %10s lba %6d%s" % (r.path, "<dir>" if r.is_dir else r.size, r.lba, extra))
    if a.extract:
        os.makedirs(a.extract, exist_ok=True)
        n = 0
        for r in d.iso.walk():
            out = os.path.join(a.extract, *r.path.strip("/").split("/"))
            if r.is_dir:
                os.makedirs(out, exist_ok=True)
            elif not d.is_cdda(r):
                with open(out, "wb") as f:
                    f.write(d.read_file(r))
                n += 1
        print("%d files extracted to %s" % (n, a.extract))
    if a.audio:
        os.makedirs(a.audio, exist_ok=True)
        for t in d.tracks:
            if t.is_audio:
                write_wav(os.path.join(a.audio, "track%02d.wav" % t.number), audio_frames(t))
        print("%d audio tracks written to %s" % (sum(t.is_audio for t in d.tracks), a.audio))


if __name__ == "__main__":
    main()

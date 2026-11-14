"""Replace a static archive from Ninja's response file.

Why: Windows caps a process command line at 32 KiB and the generated main module has 281 objects. build.py gives every
object to this script in an `llvm-ar` response file; this launcher also removes the old archive so an object removed
from build.ninja cannot remain in the library.

Usage: python tools/build/archive.py LLVM_AR OUT.a OUT.a.rsp
"""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 4:
        sys.exit(__doc__.splitlines()[-1])
    ar, archive, response = sys.argv[1:]
    Path(archive).unlink(missing_ok=True)
    subprocess.run([ar, "rcs", archive, f"@{response}"], check=True)


if __name__ == "__main__":
    main()

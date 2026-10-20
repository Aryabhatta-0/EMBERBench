"""``python -m emberbench``: the same :func:`emberbench.cli.main` as the ``emberbench`` console script."""

import os
import sys

from .cli import main

if __name__ == "__main__":
    # ``python -m`` puts the current directory first on sys.path; the console
    # script does not. Drop it so that --agent resolves the same modules from
    # both entry points and the cwd never decides which code is imported.
    if sys.path and sys.path[0] in ("", os.getcwd()):
        del sys.path[0]
    sys.exit(main())

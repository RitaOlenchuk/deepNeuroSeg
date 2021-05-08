import os
import sys
import uuid
import logging
import tempfile
import subprocess

from typing import Optional


logger = logging.getLogger(__name__)


def run_script_inline(script: str, *args, capture_output=False, check=True) -> Optional[str]:
    """
    Runs script in bash.
    """
    script_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid1()}.sh")
    with open(script_path, "w") as wfd:
        wfd.write(script)

    logger.info(script)
    output = run_script(script_path, *args, capture_output=capture_output, check=check)
    os.remove(script_path)
    return output


def run_script(path: str, *args, capture_output=False, check=True) -> Optional[str]:
    cmd = ["bash", path]
    if args:
        cmd.extend(args)

    try:
        res = subprocess.run(cmd, capture_output=capture_output, check=check)
        if capture_output:
            return res.stdout.decode("UTF-8")
        else:
            return str(res)
    except subprocess.CalledProcessError as e:
        logger.fatal(e)
        sys.exit(1)
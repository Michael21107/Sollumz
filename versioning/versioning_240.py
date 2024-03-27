"""Handle changes between 2.4.0 and 2.5.0."""

from bpy.types import (
    BlendData,
)

def do_versions(data_version: int, data: BlendData):
    if data_version < 2:
        # TODO: fix root CG offset changes!!!!!!!!!!!!1
        pass

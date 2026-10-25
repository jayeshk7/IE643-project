# applies a unified diff (the model's patch) to a program
# each hunk is found by matching its text in the program, not by its @@ line numbers,
# so a patch with slightly wrong line numbers still works. raises PatchError if a hunk's text is not in the program
#
# a hunk looks like this:
#   @@ -4,7 +4,7 @@               <- "start at old line 4", only used as a hint
#    lever_thickness = 8.0        <- " " context line: must be in the program, stays
#   -arm_depth = 10.0             <- "-" removed line: must be in the program, goes away
#   +arm_depth = 15.0             <- "+" added line
#    base_fillet_radius = 2.0

import re

class PatchError(Exception):
    pass

HEADER = re.compile(r"^@@ -(\d+)")   # the old start line in "@@ -4,7 +4,7 @@"

# splits the patch into hunks. a hunk is (start line from its @@ line or None, list of (kind, text)), kind is " ", "-" or "+"
def parse_hunks(patch):
    hunks = []
    for line in patch.split("\n"):
        if line.startswith(("---", "+++")):   # the file names, skip
            continue
        # every @@ line starts a new hunk
        if line.startswith("@@"):
            m = HEADER.match(line)
            hunks.append((int(m.group(1)) if m else None, []))
            continue
        kind = line[:1]
        # text before the first @@ is skipped, unless it already looks like a diff line (the model forgot the @@ line)
        if not hunks:
            if kind not in ("+", "-", " "):
                continue
            hunks.append((None, []))
        if kind in ("+", "-", " "):
            hunks[-1][1].append((kind, line[1:]))
        elif line == "":
            hunks[-1][1].append((" ", ""))   # an empty context line that lost its leading space
        # any other line (text the model wrote) is ignored

    # drop empty lines at the end of a hunk, and hunks that don't change anything
    out = []
    for start, body in hunks:
        while body and body[-1] == (" ", ""):
            body.pop()
        if any(kind != " " for kind, _ in body):
            out.append((start, body))
    return out

# finds where block (a list of lines) is in lines, starting at line lo
# tries an exact match first (trailing spaces ignored), then one that also ignores indentation
# if it is there more than once, takes the one closest to the @@ line number (hint)
def find_block(lines, block, hint, lo):
    n = len(block)
    for norm in (str.rstrip, str.strip):
        target = [norm(b) for b in block]
        hits = []
        for i in range(lo, len(lines) - n + 1):
            if [norm(l) for l in lines[i:i + n]] == target:
                hits.append(i)
        if hits:
            if hint is None:
                return hits[0]
            return min(hits, key=lambda i: abs(i - hint))
    return None

def apply_patch(original, patch):
    hunks = parse_hunks(patch)
    if not hunks:
        raise PatchError("no hunks")
    lines = original.split("\n")
    lo = 0      # the next hunk must come after the previous one
    shift = 0   # lines added minus lines removed so far, moves the @@ numbers of later hunks
    for start, body in hunks:
        old = [text for kind, text in body if kind in (" ", "-")]   # what the program has now
        new = [text for kind, text in body if kind in (" ", "+")]   # what it should have after
        hint = start - 1 + shift if start else None                 # 0-based line number from @@
        if old:
            pos = find_block(lines, old, hint, lo)
        elif hint is not None:   # only + lines: insert at the @@ line
            pos = hint
        else:                    # only + lines and no line number: insert before the last line
            pos = len(lines) - 1
        if pos is None:
            raise PatchError("hunk not found")
        lines[pos:pos + len(old)] = new   # swap the old lines for the new ones
        lo = pos + len(new)
        shift += len(new) - len(old)
    return "\n".join(lines)

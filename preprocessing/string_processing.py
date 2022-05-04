def ensure_path(path, terminator="/"):
    if path == None:
        return path

    path = path.replace('\\', terminator)
    if path.endswith(terminator):
        return path
    else:
        path = path + terminator
    return path

def ensure_terminator(path, terminator="/"):
    if path == None:
        return path
    if path.endswith(terminator):
        return path
    return path + terminator





    
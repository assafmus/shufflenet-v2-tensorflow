import os


def get_db(DB_PATH):
    classes = [x for x in sorted(os.listdir(DB_PATH))]
    tmp = [[os.path.join(DB_PATH, c, x) for x in sorted(os.listdir(os.path.join(DB_PATH, c)))] for c in classes]
    im_paths = []
    labels = []
    for l, t in enumerate(tmp):
        im_paths.extend(t)
        labels.extend([l] * len(t))
    return im_paths, labels, classes
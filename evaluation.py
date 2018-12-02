from imagenet40 import get_db


def evaluate_csv(csv_path):
    DB_PATH = '/data/imagenet40/val'

    data = [x.strip().split(",") for x in open(csv_path).readlines()]
    pred_im_paths = [x[0] for x in data]
    pred_labels = [int(x[1]) for x in data]
    pred_conf = [float(x[2]) for x in data]

    im_paths, labels, classes = get_db(DB_PATH)
    assert all([p in im_paths for p in pred_im_paths])
    idx = [im_paths.index(p) for p in pred_im_paths]
    pred_im_paths = [pred_im_paths[i] for i in idx]
    pred_labels = [pred_labels[i] for i in idx]
    pred_conf = [pred_conf[i] for i in idx]

    acc = sum([a==b for a, b in zip(labels, pred_labels)]) / float(len(labels))
    print "Accuracy:", acc


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_path', type=str, help='path to csv')
    args = parser.parse_args()

    evaluate_csv(csv_path=args.csv_path)

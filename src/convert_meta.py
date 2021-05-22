import click
import scipy.io
import csv
from icecream import ic


@click.command()
@click.argument("meta-file", type=click.Path(exists=True))
@click.argument("out-file", type=click.Path())
def main(meta_file, out_file):
    ic(meta_file)
    meta = scipy.io.loadmat(meta_file, squeeze_me=True)['synsets']
    ic(meta)
    id = [meta[idx][0] for idx in range(len(meta))]
    wnid = [meta[idx][1] for idx in range(len(meta))]
    words = [meta[idx][2] for idx in range(len(meta))]
    gloss = [meta[idx][3] for idx in range(len(meta))]
    with open(out_file, "w") as f:
        csv_out = csv.writer(f)
        csv_out.writerow(['id', 'wnid', 'words', 'gloss'])
        for row in zip(id, wnid, words, gloss):
            csv_out.writerow(row)


if __name__ == "__main__":
    main()
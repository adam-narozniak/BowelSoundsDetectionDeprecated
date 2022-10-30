import argparse
import csv


def convert_txt_to_csv(txt_file, csv_file):
    """Converts txt file generated from Audacity to csv file.

    Args:
        txt_file (src): Path to load txt file.
        csv_file (src): Path to save csv file.
    """
    with open(txt_file, 'r') as input, open(csv_file, 'w', newline='') as output:
        writer = csv.writer(output)
        writer.writerow(['start', 'end', 'fmin', 'fmax', 'category'])
        i = 0
        for line in input:
            i += 1
            if i % 2 == 1:
                times = line.split()
                continue
            freqs = line.split()[1:]
            if len(times) == 3:
                writer.writerow(times[:-1] + freqs + times[-1:])
            else:
                writer.writerow(times + freqs + ['NaN'])
                


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str,
                        help='Path to txt annotation file')
    args = parser.parse_args()
    output_file = args.input_file.replace('.txt', '.csv')
    convert_txt_to_csv(args.input_file, output_file)

import json
import pandas as pd
import pathlib

CUSTOM_BEHAVIOUR = True

if __name__ == '__main__':
    if CUSTOM_BEHAVIOUR:
        plots_directory = pathlib.Path('./plots')
        records_directory = pathlib.Path('./records')
        id_filename_pairs = []
        for file_path in plots_directory.iterdir():
            if file_path.is_file() and file_path.name != "BEST_RUN_SO_FAR.pdf":
                plot_filename = file_path.name
                used_id = plot_filename[35:45]
                id_filename_pairs.append((used_id, plot_filename))
        id_filename_pairs = sorted(id_filename_pairs)
        for i in range(len(id_filename_pairs)):
            upper_bound, lower_bound = id_filename_pairs[i][0], id_filename_pairs[i - 1][0] if i > 0 else "0"
            json_filenames = sorted([file.name for file in records_directory.iterdir() if upper_bound > file.name[4:14] > lower_bound])[-40:]
            all_record = []
            for json_filename in json_filenames:
                json_file = pathlib.Path(f'./records/{json_filename}')
                timestamp = json_file.name.split('.')[0].split('_')[1]
                timestamp = int(1000*float(timestamp))
                try:
                    with open(json_file, 'r') as file:
                        record = json.load(file)
                except:
                    print(json_file.name)
                    raise Exception('Loading error.')
                record['timestamp'] = timestamp
                all_record.append(record)

            df = pd.DataFrame(all_record)
            df.to_csv(f'records/good_{upper_bound}.csv', index=False)

    else:
        pathname = 'records/'
        path = pathlib.Path(pathname)

        all_record = []
        for filename in path.glob('*.json'):
            timestamp = filename.name.split('.')[0].split('_')[1]
            timestamp = int(1000*float(timestamp))
            try:
                with open(filename, 'r') as file:
                    record = json.load(file)
            except:
                print(filename.name)
                raise Exception('Loading error.')
            record['timestamp'] = timestamp
            all_record.append(record)

        df = pd.DataFrame(all_record)
        df.to_csv(f'{pathname}/records.csv', index=False)
import json
import pandas as pd
import pathlib

if __name__ == '__main__':
    path = pathlib.Path('records/')

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
    df.to_csv('records.csv', index=False)
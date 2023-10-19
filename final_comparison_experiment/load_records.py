import json
import pandas as pd
import pathlib

from matplotlib import pyplot as plt


def create_csvs_from_plots():
    plots_directory = pathlib.Path('./plots')
    records_directory = pathlib.Path('./records')
    id_filename_pairs = []
    for file_path in plots_directory.iterdir():
        if file_path.is_file() and file_path.name != "BEST_RUN_SO_FAR.pdf":
            plot_filename = file_path.name
            used_id = plot_filename[35:45]
            id_filename_pairs.append((used_id, plot_filename))
    id_filename_pairs = sorted(id_filename_pairs)
    for used_id, plot_filename in id_filename_pairs:
        json_filenames = sorted([file.name for file in records_directory.iterdir() if "200000000000000000" > file.name[4:14] > used_id])[:40]
        all_record = []
        for json_filename in json_filenames:
            json_file = pathlib.Path(f'./records/{json_filename}')
            timestamp = json_file.name.split('.')[0].split('_')[1]
            timestamp = int(1000 * float(timestamp))
            try:
                with open(json_file, 'r') as file:
                    record = json.load(file)
            except:
                print(json_file.name)
                raise Exception('Loading error.')
            record['timestamp'] = timestamp
            all_record.append(record)

        df = pd.DataFrame(all_record)
        df.to_csv(f'records/good_{used_id}.csv', index=False)
        print(plot_filename)
        print(df.iloc[0], df.iloc[-1])

def create_all_csvs():
    pathname = 'records/'
    path = pathlib.Path(pathname)

    all_record = []
    for filename in path.glob('*.json'):
        timestamp = filename.name.split('.')[0].split('_')[1]
        timestamp = int(1000 * float(timestamp))
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


def create_plots_from_csvs():
    current_directory = pathlib.Path()
    csv_files = [file for file in current_directory.glob('./records/*good*.csv')]
    for csv_file in csv_files:
        data = pd.read_csv(csv_file)
        alpha_kappas, beta_kappas = [], []
        alpha_objectives, beta_objectives = [], []
        alpha_nd_losses_00, alpha_nd_losses_01, beta_nd_losses_00, beta_nd_losses_01 = [], [], [], []
        for index, row in data.iterrows():
            kappa = max([row['alpha_loss_test_00'], row['alpha_loss_test_01'], row['alpha_loss_test_10'], row['alpha_loss_test_11']])
            objective = row['objective_loss_test']
            nd_loss_00, nd_loss_01 = row['alpha_loss_test_00'], row['alpha_loss_test_01']
            if row['fairness_name'] == 'Alpha':
                alpha_kappas.append(kappa)
                alpha_objectives.append(objective)
                alpha_nd_losses_00.append(nd_loss_00)
                alpha_nd_losses_01.append(nd_loss_01)
            else:
                beta_kappas.append(kappa)
                beta_objectives.append(objective)
                beta_nd_losses_00.append(nd_loss_00)
                beta_nd_losses_01.append(nd_loss_01)

        plt.scatter(alpha_kappas, alpha_objectives, c='blue',label='alpha', marker='v')
        plt.scatter(beta_kappas, beta_objectives, c='blue',label='beta', marker='x')
        plt.xlabel("kappa")
        plt.ylabel("test objective loss")
        plt.legend()
        title_topics = ['eta', 'gamma_0', 'gamma_1', 'information_0', 'information_1', 'feature_size_0', 'feature_size_1']
        title_values = [str(data[topic][0]) for topic in title_topics]
        title = '_'.join(title_values) + '_CON'
        plt.title(title)
        plt.savefig(f'./plots/{csv_file.name[5:15]}_what_we_suffer.pdf')
        plt.clf()

        plt.scatter(alpha_kappas, alpha_nd_losses_00, c='blue', label='alpha', marker='v')
        plt.scatter(beta_kappas, beta_nd_losses_00, c='blue', label='beta', marker='x')
        plt.xlabel("kappa")
        plt.ylabel("alpha loss on Y=A=0")
        plt.legend()
        title_topics = ['eta', 'gamma_0', 'gamma_1', 'information_0', 'information_1', 'feature_size_0',
                        'feature_size_1']
        title_values = [str(data[topic][0]) for topic in title_topics]
        title = '_'.join(title_values) + '_PRO_00'
        plt.title(title)
        plt.savefig(f'./plots/{csv_file.name[5:15]}_what_we_gain_00.pdf')
        plt.clf()

        plt.scatter(alpha_kappas, alpha_nd_losses_01, c='blue', label='alpha', marker='v')
        plt.scatter(beta_kappas, beta_nd_losses_01, c='blue', label='beta', marker='x')
        plt.xlabel("kappa")
        plt.ylabel("alpha loss on Y=1, A=0")
        plt.legend()
        title_topics = ['eta', 'gamma_0', 'gamma_1', 'information_0', 'information_1', 'feature_size_0',
                        'feature_size_1']
        title_values = [str(data[topic][0]) for topic in title_topics]
        title = '_'.join(title_values) + '_PRO_01'
        plt.title(title)
        plt.savefig(f'./plots/{csv_file.name[5:15]}_what_we_gain_01.pdf')
        plt.clf()


if __name__ == '__main__':
    create_plots_from_csvs()
    # create_csvs_from_plots()
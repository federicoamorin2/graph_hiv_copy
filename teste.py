from datetime import datetime, timedelta
from random import randrange
import pandas as pd
import numpy as np
from tqdm import tqdm

def random_date(start, end):
    """
    This function will return a random datetime between two datetime 
    objects.
    """
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = randrange(int_delta)
    return (start + timedelta(seconds=random_second)).strftime("%Y-%m-%d")

d1 = datetime.strptime('2018-01-01', "%Y-%m-%d")
d2 = datetime.strptime('2023-02-01', "%Y-%m-%d")
d3 = datetime.now()
df = pd.DataFrame(
    {
        "dtent": [random_date(d1, d2) for _ in range(100000)]
    }
)

df["dtenc"] = df["dtent"].apply(lambda x: random_date(datetime.strptime(x, "%Y-%m-%d"), d3))
df['dtent'] = pd.to_datetime(df['dtent'])
df['dtenc'] = pd.to_datetime(df['dtenc'])
df['idade'] = ((df['dtenc'] - df['dtent'])/np.timedelta64(1, 'M')).astype(int)
df["pasta"] = df.index


train_size = 2000
test_size = 1000
valid_size = 1000

def update_picks(df_of_intrest, train_picks, test_picks, valid_picks, train_size, test_size, valid_size, pct):
    train_samples = train_size * pct
    test_samples = test_size * pct
    valid_samples = valid_size * pct
    counter = 0
    while train_samples > 0 or test_samples > 0 or valid_samples > 0:
        counter += 1
        current_sample = df_of_intrest[df_of_intrest['group'] == counter]
        current_picks = []
        if train_samples > 0:
            train_pick = current_sample.sample(1).pasta.item()
            current_picks.append(train_pick)
            train_picks.append(train_pick)
            train_samples -= 1

        if test_samples > 0:
            test_pick = current_sample[~current_sample.pasta.isin(current_picks)].sample(1).pasta.item()
            current_picks.append(test_pick)
            test_picks.append(test_pick)
            test_samples -= 1
        if valid_samples > 0:
            valid_pick = current_sample[~current_sample.pasta.isin(current_picks)].sample(1).pasta.item()
            valid_picks.append(valid_pick)
            valid_samples -= 1
        
    return train_picks, test_picks, valid_picks

def split_data(df, baseline_df, train_size, test_size, valid_size):
    train_picks, test_picks, valid_picks = [], [], []
    month_distribution = baseline_df.idade.value_counts(normalize=True).reset_index()
    for _, idade, pct in tqdm(month_distribution.itertuples(), total=month_distribution.shape[0]):
        df_of_intrest = df[df.idade == idade]
        df_of_intrest = df_of_intrest.sort_values('dtent', ascending=False)
        df_of_intrest['group'] = df_of_intrest.reset_index(drop=True).index // 3 + 1
        train_picks, test_picks, valid_picks = update_picks(df_of_intrest, train_picks, test_picks, valid_picks, train_size, test_size, valid_size, pct)
    
    return train_picks, test_picks, valid_picks

train_picks, test_picks, valid_picks = split_data(df, df, train_size, test_size, valid_size)

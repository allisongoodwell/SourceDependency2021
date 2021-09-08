import numpy as np
import random
from scipy.stats import percentileofscore



def cal_nse(obs, mod):
    mo = np.nanmean(obs)
    a = np.nansum([(mi - oi) ** 2 for mi, oi in zip(mod, obs)])
    b = np.nansum([(oi - mo) ** 2 for oi in obs])
    
    return 1 - a / b


def cal_mape(obs, mod):
    mo = np.nanmean(obs)
    ape = [np.abs(mi - oi) / mo for mi, oi in zip(mod, obs)]
    
    return np.nanmean(ape)


def shannon_entropy(x, bins):
    c = np.histogramdd(x, bins)[0]
    p = c / np.sum(c)
    p = p[p > 0]
    h =  - np.sum(p * np.log2(p))
    
    return h


def interaction_information(mi_c, mi):
    i = mi_c - mi
    return i


def normalized_source_dependency(mi_s1_s2, H_s1, H_s2):
    i = mi_s1_s2 / np.min([H_s1, H_s2])
    
    return i


def redundant_information_bounds(mi_s1_tar, mi_s2_tar, interaction_info):
    r_mmi = np.min([mi_s1_tar, mi_s2_tar])
    r_min = np.max([0, - interaction_info])
    
    return r_mmi, r_min


def rescaled_redundant_information(mi_s1_s2, H_s1, H_s2, mi_s1_tar, mi_s2_tar, interaction_info):
    norm_s_dependency = normalized_source_dependency(mi_s1_s2, H_s1, H_s2)
    r_mmi, r_min = redundant_information_bounds(mi_s1_tar, mi_s2_tar, interaction_info)
    
    return r_min + norm_s_dependency * (r_mmi - r_min)


def mutual_information(dfi, source, target, bins, reshuffle=0):
    x = dfi[source].values
    y = dfi[target].values
    if reshuffle == 1:
        random.shuffle(x)
        random.shuffle(y)
    H_x = shannon_entropy([x], [bins])
    H_y = shannon_entropy([y], [bins])
    H_xy = shannon_entropy([x, y], [bins, bins])
    
    return H_x + H_y - H_xy


def conditional_mutual_information(dfi, source, target, condition, bins, reshuffle=0):
    x = dfi[source].values
    y = dfi[condition].values
    z = dfi[target].values
    if reshuffle == 1:
        random.shuffle(x)
        random.shuffle(y)
        random.shuffle(z)

    H_y = shannon_entropy([y],[bins])
    H_z = shannon_entropy([z],[bins])
    H_xy = shannon_entropy([x, y],[bins, bins])
    H_zy = shannon_entropy([z, y],[bins, bins])
    H_xyz = shannon_entropy([x, y, z],[bins, bins, bins])
    
    return H_xy + H_zy - H_y - H_xyz


def information_partitioning(df, source_1, source_2, target, bins, reshuffle=0):
    if reshuffle == 1:
        x = df[source_1].values
        y = df[source_2].values
        z = df[target].values
        random.shuffle(x)
        random.shuffle(y)
        random.shuffle(z)
        df['source_1'] = x
        df['source_2'] = y
        df['target'] = z
    else:
        df['source_1'] = df[source_1].values
        df['source_2'] = df[source_2].values
        df['target'] = df[target].values

    H_s1 = shannon_entropy(df['source_1'].values, [bins])
    H_s2 = shannon_entropy(df['source_2'].values, [bins])
    mi_s1_s2 = mutual_information(df, 'source_1', 'source_2', bins)
    mi_s1_tar = mutual_information(df, 'source_1', 'target', bins)
    mi_s2_tar = mutual_information(df, 'source_2', 'target', bins)
    mi_s1_tar_cs2 = conditional_mutual_information(df, 'source_1', 'target', 'source_2', bins)
    interaction_info = interaction_information(mi_s1_tar_cs2, mi_s1_tar)

    redundant = rescaled_redundant_information(mi_s1_s2, H_s1, H_s2, mi_s1_tar, mi_s2_tar, interaction_info)
    unique_s1 = mi_s1_tar - redundant
    unique_s2 = mi_s2_tar - redundant
    synergistic = interaction_info + redundant
    total_information = unique_s1 + unique_s2 + redundant + synergistic
    
    return total_information, unique_s1, unique_s2, redundant, synergistic


def sig_test_info_partitioning(df, source_1, source_2, target, bins, nshuffles=1000):
    df = df[[source_1, source_2, target]].dropna()
    H_tar = shannon_entropy(df[target].values, [bins])
    I_tar_s1 = mutual_information(df, source_1, target, bins, reshuffle=0)
    I_tar_s2 = mutual_information(df, source_2, target, bins, reshuffle=0) 
    total_information, unique_1, unique_2, redundant, synergistic = information_partitioning(df, source_1, source_2, target, bins, reshuffle=0)
    info_bootstrapping = [information_partitioning(df, source_1, source_2, target, bins, reshuffle=1) for i in range(nshuffles)]
    total_info_bootstrapping = zip(*info_bootstrapping)[0]
    
    return [total_information, unique_1, unique_2, synergistic, redundant, H_tar, I_tar_s1, I_tar_s2, \
            percentileofscore(total_info_bootstrapping, total_information) / 100.]


def sig_test_conditional_mutual_information(df, source, target, condition, bins, nshuffles=1000):
    df = df[[source, target, condition]].dropna()
    cmi = conditional_mutual_information(df, source, target, condition, bins, reshuffle=0)
    cmi_bootstrapping = [conditional_mutual_information(df, source, target, condition, bins, reshuffle=1) for i in range(nshuffles)]
    
    return cmi, percentileofscore(cmi_bootstrapping, cmi) / 100.


def sig_test_mutual_information(df, source, target, bins, nshuffles=1000):
    df = df[[source, target]].dropna()
    mi = mutual_information(df, source, target, bins, reshuffle=0)
    mi_bootstrapping = [mutual_information(df, source, target, bins, reshuffle=1) for i in range(nshuffles)]
    
    return mi, percentileofscore(mi_bootstrapping, mi) / 100.


if __name__ == "__main__":

    pass


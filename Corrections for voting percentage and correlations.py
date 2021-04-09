
import numpy as np
import pandas as pd
import matplotlib as plt
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
#todo in the plot: qj_hat equals to q (orange and red) - why?
NUM_OF_BIG_PARTIES = 8

NUM_OF_CITIES = 1213

PARTY_NAMES = ["טב", "ל", "מחל", "שס", "ג", "פה", "אמת", "ודעם"][::-1]
NUM_OF_KALPIES = 10631
SIM_ITER_NUM = 50
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

def party_bar(pj, qj_hat, true_qj, title='' , xlabel='' , ylabel='' , names=[],vtype=""):
    width = 0.3  # set column width
    fig, ax = plt.subplots()  # plt.subplots()

    pj_bar = ax.bar(np.arange(8), list(pj), width, color='b',align='center', alpha=0.5, ecolor='black', capsize=10)
    qj_hat_bar=ax.bar(np.arange(8)+width+width, list(qj_hat), width, color='g',align='center', alpha=0.5, ecolor='black', capsize=10)
    p_bar = ax.bar(np.arange(8)+width, list(true_qj), width, color='r')
    ax.set_ylabel('Votes percent')
    ax.set_xlabel('Parties Names')
    ax.set_title('Comparing the voting  precentage per parties\n between the simulation , true distribtion and the estimator\n for the true distribution , under the vector : '+vtype   )
    ax.set_xticks(np.arange(len(names)))
    rev_names = [name[::-1] for name in list(names)]
    long_names=[]
    names_dict={"פה":"כחול לבן","ג":"יהדות התורה", "שס":"שס","מחל":"ליכוד","ל":"ישראל ביתנו","טב":"ימינה","אמת":"עבודה גשר מרץ","ודעם":"הרשימה המשותפת"}

    for name in rev_names:
        long_names.append(names_dict[name[::-1]][::-1])

    ax.set_xticklabels(long_names, rotation=45)
    dummy_1 = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    dummy_2 = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    ax.legend((pj_bar[0], qj_hat_bar[0],p_bar[0]), ("p","q hat", "q"))
    plt.show()
    return fig, ax

def clear_data():
    """
    clear the un necessary data from file
    :return:
    """
    # data process
    df_2020_raw = pd.read_csv("vote_per_kalpi_2020.csv",
                              encoding='iso-8859-8',
                              index_col='שם ישוב')
    # clear the data
    df_2020_raw = df_2020_raw.drop('מעטפות חיצוניות', axis=0)
    df_2020 = df_2020_raw.drop('סמל ועדה', axis=1)  # new column added in Sep 2019
    df_2020 = df_2020[df_2020.columns[9:]]
    df_2020_raw.drop("זץ", axis=1)
    df_2020.drop("זץ", axis=1)
    return df_2020_raw, df_2020

def create_q_vector(df, thresh=0.0325):
    df = df[PARTY_NAMES]
    return df  # the biggest 8 of the list

def create_v1(data_frame, data_frame_raw):
    n_tilda = data_frame * (data_frame_raw['בזב'].sum() / data_frame_raw['כשרים'].sum())  # calc like we asked to
    n_tilda = n_tilda.round()  # round it
    div = (data_frame_raw['כשרים'] / data_frame_raw['בזב'])
    trans = (np.array(div)).reshape(NUM_OF_KALPIES, 1)  # transpose to vector
    matrix_of_ones = np.ones((1, 8))
    v1 = trans @ matrix_of_ones  # ( (10631, 1) * (1,8) -> (10631,8))
    return n_tilda, v1


def lab_3_question_two(data_frame_raw: pd.DataFrame, data_frame: pd.DataFrame):
    """
    :param data_frame: our df
    :return:
    """

    n_tilda, v1 = create_v1(data_frame, data_frame_raw)


    # part 2.B - score_vector:
    score_vector = np.array([0.5, 0.4, 0.7, 0.8, 0.9, 0.6, 0.3, 0.2])
    # we chose each prob to party by our knowledge of the political situation
    big_matrix_of_ones = np.ones((1, NUM_OF_KALPIES))
    v2 = (score_vector.reshape((8, 1))) @ big_matrix_of_ones
    v2 = v2.T

    # part 2.C - random_vector on each city:
    # create matrix of (1213,8) that each row is permutation of 0.2 until 0.9
    matrix = np.zeros(shape=(NUM_OF_CITIES, 8)) + score_vector
    array = np.apply_along_axis(arr=matrix, axis=1, func1d=np.random.permutation)

    # create dictionary that matches each random_vector to a city
    names_of_cities = set(list(data_frame.index))
    # print(names_of_cities)
    mat_of_permutation = pd.DataFrame(array, index=names_of_cities)  # matrix of cities and vectors
    mat_of_permutation_trans = mat_of_permutation.T
    n_tilda_trans = n_tilda.T

    lista = []

    run_to = data_frame_raw.index
    run_to = run_to.drop_duplicates()  # each city in israel
    # print(mat_of_permutation_trans.shape)
    # print(n_tilda_trans.shape)
    # print(len(run_to))
    for name in run_to:
        mat_np_trans = np.array(mat_of_permutation_trans[name]).reshape((8, 1))  # random vector after permutation
        calc_np_trans = np.array(n_tilda_trans[name])  # num of votes for each party from some city

        if calc_np_trans.shape == (8,):  # If a row vector is returned broadcasting will create a matrix instead of vec.
            shape_calc = np.zeros_like(calc_np_trans)  # create matrix os zero's like calc_np_trans
            shape_calc = shape_calc.reshape((8, 1))
            shape_calc = shape_calc + mat_np_trans
            lista.append(shape_calc.T)
            continue

        shape_calc = np.zeros_like(calc_np_trans)  # create matrix os zero's like calc_np_trans
        calc_np_trans = shape_calc + mat_np_trans  # add to each raw the vector
        lista.append(calc_np_trans.T)  # append to list after Transpose

    tuple_of_lista = np.vstack(tuple(lista))
    v3 = pd.DataFrame(tuple_of_lista, index=n_tilda.index, columns=n_tilda.columns)
    # print(final.T['כפר חיטים'].shape)
    # and so far we create V_i

    return v1, v2, np.array(v3), n_tilda







def main():

    df_raw_top, df = clear_data()
    #print(df_raw_top[["בזב"]].shape)
    df = create_q_vector(df)  # top 8

    # Question 2:
    #lab_4_Q2(df, df_raw_top)

    lab_4_Q3()


def lab_4_Q2(df, df_raw_top):
    q, ps_averages, ps_stds, qs_hat_averages, qs_hat_stds, vs_qs_hat_ols_avg, vs_qs_hat_ols_std = extract_data_measures(
        df,
        df_raw_top)
    for i in range(3):
        plot_bars_means(q, qs_hat_averages[i], qs_hat_stds[i]
                        , ps_averages[i], ps_stds[i], vs_qs_hat_ols_avg[i], vs_qs_hat_ols_std[i], simu_kind="aa")


def extract_data_measures(df, df_raw_top):
    v1, v2, v3, n_tilda = lab_3_question_two(df_raw_top, df)
    vs = [v1, v2, v3]
    q = n_tilda.sum(axis=0) / n_tilda.sum().sum()
    simulations = [np.zeros((NUM_OF_KALPIES, NUM_OF_BIG_PARTIES))
        , np.zeros((NUM_OF_KALPIES, NUM_OF_BIG_PARTIES))
        , np.zeros((NUM_OF_KALPIES, NUM_OF_BIG_PARTIES))]
    vs_qs_hat_sim = [[], [], []]
    vs_ps_sim = [[], [], []]
    vs_qs_hat_ols = [[], [], []]
    Y = (n_tilda.sum(axis=1))
    Y = np.array(Y).reshape(NUM_OF_KALPIES, 1)
    for i in range(SIM_ITER_NUM - 1):
        for j in range(len(simulations)):
            cur_sim = np.random.binomial(np.array(n_tilda, dtype=np.int32), (vs[j]))
            sim_vs_j = cur_sim / vs[j]
            cur_non_potential_votes_to_party = cur_sim.sum(axis=0)
            cur_non_potential_votes = cur_sim.sum().sum()
            cur_potential_votes_to_party = sim_vs_j.sum(axis=0)
            cur_potential_votes = sim_vs_j.sum().sum()
            vs_qs_hat_ols[j].append(q1_lab4(cur_sim, Y))
            vs_qs_hat_sim[j].append(cur_potential_votes_to_party / cur_potential_votes)
            vs_ps_sim[j].append(cur_non_potential_votes_to_party / cur_non_potential_votes)
    vs_qs_hat_sim = np.array(vs_qs_hat_sim)
    vs_ps_sim = np.array(vs_ps_sim)
    vs_qs_hat_ols = np.array(vs_qs_hat_ols)
    qs_hat_stds = vs_qs_hat_sim.std(axis=1)
    ps_stds = vs_ps_sim.std(axis=1)
    qs_hat_averages = vs_qs_hat_sim.mean(axis=1)
    ps_averages = vs_ps_sim.mean(axis=1)
    vs_qs_hat_ols_avg = vs_qs_hat_ols.std(axis=1)
    vs_qs_hat_ols_std = vs_qs_hat_ols.mean(axis=1)
    return q, ps_averages, ps_stds, qs_hat_averages, qs_hat_stds, vs_qs_hat_ols_avg, vs_qs_hat_ols_std


from matplotlib import pyplot as plt


def plot_bars_means(q_mean, qj_mean_v1, qj_var_v1, pj_mean_v1, pj_var_v1, qs_hat_std_ols, qs_hat_avg_ols, simu_kind=""):
  x_pos = 1.5 * np.arange(8)
  print(qs_hat_avg_ols)
  print(qs_hat_std_ols)
  width=0.25
  bar_width=0.25
  fig, ax = plt.subplots()
  bar_ols=ax.bar(x_pos, np.array(qs_hat_avg_ols), yerr=np.array(qs_hat_std_ols), width=bar_width,align='center', alpha=0.5, ecolor='black', capsize=5)
  bar_qj=ax.bar(x_pos+width+width, np.array(qj_mean_v1), yerr=np.array(qj_var_v1),width=bar_width, align='center', alpha=0.5, ecolor='black', capsize=5)
  bar_pj=ax.bar(x_pos+width, np.array(pj_mean_v1), yerr=np.array(pj_var_v1), width=bar_width,align='center', alpha=0.5, ecolor='black', capsize=5)
  bar_q=ax.bar(x_pos+width+width+width, np.array(q_mean), width=bar_width, align='center', alpha=0.5, ecolor='black', capsize=5)
  ax.set_title('Compering mean and variance between the following simulation : '+ simu_kind)
  ax.set_xticks(x_pos)
  names=PARTY_NAMES
  rev_names = [name[::-1] for name in list(names)]
  long_names=[]
  dict = {"פה": "כחול לבן", "ג": "יהדות התורה", "שס": "שס", "מחל": "ליכוד", "ל": "ישראל ביתנו", "טב": "ימינה",
          "אמת": "עבודה גשר מרץ", "ודעם": "הרשימה המשותפת"}

  for name in rev_names:
      long_names.append(dict[name[::-1]][::-1])
  ax.set_xticklabels(long_names,rotation=45)
  ax.set_ylabel('vote precentange ')

  ax.yaxis.grid(True)
  dummy_1 = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
  dummy_2 = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
  ax.legend((bar_qj[0], bar_pj[0],bar_ols[0], bar_q), ("qj_hat","pj","qj_ols_hat", "q"))

  # Save the figure and show
  plt.tight_layout()
  plt.savefig('bar_plot_with_error_bars.png')
  plt.show()

stim_names=["\n,  Vi is the active vote percentage  in ballot i  in the original data\n","\n Every party  is assigned  a different scalar"
                                          " from the set [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]"
                                          "\nand that will define its active vote % over all the ballots.\n","\n : Every ballot gets a random permutation from the set [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]\n"
                                          "Which define each party active % vote for the current ballot.\n"]


def lab_4_Q3():
    df_raw = pd.read_csv("votes per city 2020.csv", encoding='iso-8859-8')
    eshkol_hevrati_calcali = pd.read_csv("eshkol_hevrati_calcali.csv", encoding='iso-8859-8', index_col='set_code')

    print(eshkol_hevrati_calcali)

    merged = df_raw.join(eshkol_hevrati_calcali, on='סמל ישוב')
    merged = merged.dropna(axis=0)  # by row
    print(merged)

    #print(new['חיפה'])




def q1_lab4(X, Y):
    model = OLS(Y, X)  # linear reg with c lines and k variables
    result = model.fit()  # fit the model


    # Question 1.2:
    # q_j = vector 8th len - the prob to vote per party in israel (if ***everybody would vote***)

    potential_per_party = (X * result.params).sum(axis=0)
    total_potential = potential_per_party.sum()  # sum of sum
    q_j_hat = potential_per_party / total_potential  # ratio like we saw in cass
    return q_j_hat


if __name__== "__main__":
    main()

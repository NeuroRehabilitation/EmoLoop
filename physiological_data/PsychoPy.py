from csv_load import *

def Horror(data):
    horror = {}
    for users in sorted(data.keys()):
        x = pd.DataFrame(columns=["videos"])
        for index, row in data[users].iterrows():
            if ('Horror' in row['videos']):
                d = {"videos": row['videos'], 'valence': row['slider_valence.response'],
                     'arousal': row['slider_arousal.response'],
                     'dominance': row['slider_dominance.response'], 'anticipatory': row['anticipatory_slider.response'],
                     'perceived_control': row['perceived_obstacles_slider.response'],
                     'novelty': row['novelty.response'],
                     'unexpectedness': row['unexpectedness.response'], 'intrisic_goal': row['intrisic_goal.response'],
                     'perceived_obstacles': row['perceived_obstacles.response'],
                     'control_stress': row['control_stress.response']}
                x = x.append(d, ignore_index=True)
        horror[users] = x

        for users in horror.keys():
            horror[users] = horror[users].sort_values('videos')

    return horror

def Erotic(data):
    erotic = {}
    for users in sorted(data.keys()):
        x = pd.DataFrame(columns=["videos"])
        for index, row in data[users].iterrows():
            if ('Erotic' in row['videos']):
                d = {"videos": row['videos'], 'valence': row['slider_valence.response'],
                     'arousal': row['slider_arousal.response'],
                     'dominance': row['slider_dominance.response'], 'anticipatory': row['anticipatory_slider.response'],
                     'perceived_control': row['perceived_obstacles_slider.response'],
                     'novelty': row['novelty.response'],
                     'unexpectedness': row['unexpectedness.response'], 'intrisic_goal': row['intrisic_goal.response'],
                     'perceived_obstacles': row['perceived_obstacles.response'],
                     'control_stress': row['control_stress.response']}
                x = x.append(d, ignore_index=True)
        erotic[users] = x

        for users in erotic.keys():
            erotic[users] = erotic[users].sort_values('videos')

    return erotic

def Scenery(data):
    scenery = {}
    for users in sorted(data.keys()):
        x = pd.DataFrame(columns=["videos"])
        for index, row in data[users].iterrows():
            if ('Scenery' in row['videos']):
                d = {"videos": row['videos'], 'valence': row['slider_valence.response'],
                     'arousal': row['slider_arousal.response'],
                     'dominance': row['slider_dominance.response'], 'anticipatory': row['anticipatory_slider.response'],
                     'perceived_control': row['perceived_obstacles_slider.response'],
                     'novelty': row['novelty.response'],
                     'unexpectedness': row['unexpectedness.response'], 'intrisic_goal': row['intrisic_goal.response'],
                     'perceived_obstacles': row['perceived_obstacles.response'],
                     'control_stress': row['control_stress.response']}
                x = x.append(d, ignore_index=True)
        scenery[users] = x

        for users in scenery.keys():
            scenery[users] = scenery[users].sort_values('videos')

    return scenery

def Social_positive(data):
    social_positive = {}
    for users in sorted(data.keys()):
        x = pd.DataFrame(columns=["videos"])
        for index, row in data[users].iterrows():
            if ('Social Positive' in row['videos']):
                d = {"videos": row['videos'], 'valence': row['slider_valence.response'],
                     'arousal': row['slider_arousal.response'],
                     'dominance': row['slider_dominance.response'], 'anticipatory': row['anticipatory_slider.response'],
                     'perceived_control': row['perceived_obstacles_slider.response'],
                     'novelty': row['novelty.response'],
                     'unexpectedness': row['unexpectedness.response'], 'intrisic_goal': row['intrisic_goal.response'],
                     'perceived_obstacles': row['perceived_obstacles.response'],
                     'control_stress': row['control_stress.response']}
                x = x.append(d, ignore_index=True)
        social_positive[users] = x

        for users in social_positive.keys():
            social_positive[users] = social_positive[users].sort_values('videos')

    return social_positive

def Social_negative(data):
    social_negative = {}
    for users in sorted(data.keys()):
        x = pd.DataFrame(columns=["videos"])
        for index, row in data[users].iterrows():
            if ('Social Negative' in row['videos']):
                d = {"videos": row['videos'], 'valence': row['slider_valence.response'],
                     'arousal': row['slider_arousal.response'],
                     'dominance': row['slider_dominance.response'], 'anticipatory': row['anticipatory_slider.response'],
                     'perceived_control': row['perceived_obstacles_slider.response'],
                     'novelty': row['novelty.response'],
                     'unexpectedness': row['unexpectedness.response'], 'intrisic_goal': row['intrisic_goal.response'],
                     'perceived_obstacles': row['perceived_obstacles.response'],
                     'control_stress': row['control_stress.response']}
                x = x.append(d, ignore_index=True)
        social_negative[users] = x

        for users in social_negative.keys():
            social_negative[users] = social_negative[users].sort_values('videos')

    return social_negative

def Category_Dataframe(category,dimension):

    df1 = pd.DataFrame(columns=[category["P10_S1"]["videos"]])
    df2 = pd.DataFrame(columns=[category["P10_S2"]["videos"]])
    users_list = list()

    for users in category.keys():
        if ("EMDB/A" in str(category[users]["videos"])):
            users_list.append(users.split("_")[0])
            df1.loc[len(df1)] = category[users][dimension].values
        if ("EMDB/B" in str(category[users]["videos"])):
            df2.loc[len(df2)] = category[users][dimension].values
    df1.insert(0, "Users", users_list)
    df = (df1.join(df2)).set_index("Users")

    return df

def Full_Dataframe(data,dimension):
    horror = Horror(data)
    erotic = Erotic(data)
    scenery = Scenery(data)
    social_positive = Social_positive(data)
    social_negative = Social_negative(data)

    df_erotic = Category_Dataframe(erotic,dimension)
    df_horror = Category_Dataframe(horror,dimension)
    df_scenery = Category_Dataframe(scenery,dimension)
    df_positive = Category_Dataframe(social_positive,dimension)
    df_negative = Category_Dataframe(social_negative,dimension)

    df = (((df_erotic.join(df_horror)).join(df_negative)).join(df_scenery)).join(df_positive)

    return df

# path = 'G:\\O meu disco\\PhD\\1st Study\\PsychoPy Data\\'
# data = CSV_Load(path)
# horror = Horror(data)


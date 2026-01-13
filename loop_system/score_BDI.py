import pandas as pd

# Load Excel file
df = pd.read_excel(r"C:\Users\Rodrigo\Downloads\Questionário Online(1-41).xlsx")

# Define scoring system
scoring = {
    "2": {
        "Não me sinto triste.": 0,
        "Sinto-me triste muitas vezes.": 1,
        "Sinto-me sempre triste.": 2,
        "Estou tão triste ou infeliz que já não aguento.": 3,
    },
    "3": {
        "Não me sinto desencorajado em relação ao futuro.": 0,
        "Sinto-me mais desencorajado em relação ao futuro do que antes.": 1,
        "Já não espero que os meus problemas se resolvam.": 2,
        "Não tenho qualquer esperança no futuro; tudo só pode piorar.": 3,
    },
    "4": {
        "Não me considero um falhado.": 0,
        "Fracassei mais vezes do que deveria.": 1,
        "Revendo o passado, o que noto é uma quantidade de fracassos.": 2,
        "Sinto-me completamente falhado como pessoa.": 3,
    },
    "5": {
        "Tenho tanto prazer como antes com as coisas que eu gosto.": 0,
        "Eu não gosto tanto das coisas como costumava.": 1,
        "Tenho pouco prazer com as coisas que eu costumava gostar.": 2,
        "Não tenho qualquer prazer nas coisas que costumava gostar.": 3,
    },
    "6": {
        "Não me sinto particularmente culpado.": 0,
        "Sinto-me culpado por muitas coisas que fiz ou devia ter feito.": 1,
        "Sinto-me bastante culpado a maioria das vezes.": 2,
        "Sinto-me culpado durante o tempo todo.": 3,
    },
    "7": {
        "Não me sinto que esteja a ser castigado.": 0,
        "Sinto que posso vir a ser castigado.": 1,
        "Espero vir a ser castigado.": 2,
        "Sinto que estou a ser castigado.": 3,
    },
    "8": {
        "Aquilo que acho de mim é o que sempre achei.": 0,
        "Perdi confiança em mim próprio.": 1,
        "Estou desapontado comigo mesmo.": 2,
        "Eu não gosto de mim.": 3,
    },
    "9": {
        "Não me critico mais que o habitual.": 0,
        "Critico-me mais do que costumava.": 1,
        "Critico-me por todas as minhas falhas.": 2,
        "Culpo-me de tudo o que de mal me acontece.": 3,
    },
    "10": {
        "Não tenho qualquer ideia de me matar.": 0,
        "Tenho ideias de me matar mas não as levarei a cabo.": 1,
        "Gostaria de me matar.": 2,
        "Matar-me-ia se tivesse a oportunidade.": 3,
    },
    "11": {
        "Não choro mais do que costumava.": 0,
        "Choro mais do que costumava.": 1,
        "Choro por tudo e por nada.": 2,
        "Apetece-me chorar, mas já não consigo.": 3,
    },
    "12": {
        "Não me sinto mais inquieto.": 0,
        "Sinto-me mais inquieto que o habitual.": 1,
        "Estou tão agitado que é difícil parar quieto.": 2,
        "Estou tão agitado que tenho de me manter a fazer algo.": 3,
    },
    "13": {
        "Não perdi o interesse nos outros ou nas minhas atividades.": 0,
        "Estou menos interessado pelas ou pelas outras pessoas.": 1,
        "Perdi a maioria do interesse nas coisas e nas outras pessoas.": 2,
        "É difícil interessar-me por qualquer coisa que seja.": 3,
    },
    "14": {
        "Tomo decisões como sempre o fiz.": 0,
        "Acho mais difícil tomar decisões do que o habitual.": 1,
        "É muito mais difícil tomar decisões do que o habitual.": 2,
        "Sinto-me incapaz de tomar qualquer decisão.": 3,
    },
    "15": {
        "Não me considero incapaz/inútil.": 0,
        "Não me considero tão válido e útil como costumava.": 1,
        "Sinto-me mais inútil, em relação às outras pessoas.": 2,
        "Sinto-me completamente inútil.": 3,
    },
    "16": {
        "Tenho a mesma energia de sempre.": 0,
        "Sinto-me com menos energia do que o habitual.": 1,
        "Não me sinto com energia para muitas coisas.": 2,
        "Não me sinto com energia para nada.": 3,
    },
    "17": {
        "Não notei qualquer mudança no meu sono.": 0,
        "Durmo um pouco mais do que o habitual.": 1,
        "Durmo um pouco menos que o habitual.": 1,
        "Durmo muito mais do que o habitual.": 2,
        "Durmo muito menos que o habitual.": 2,
        "Durmo a maior parte do tempo durante o dia.": 3,
        "Acordo 1-2 horas mais cedo e não consigo voltar a dormir.": 3,
    },
    "18": {
        "Não estou mais irritável que o normal.": 0,
        "Estou mais irritável do que o habitual.": 1,
        "Estou muito mais irritável que o normal.": 2,
        "Estou irritável o tempo todo.": 3,
    },
    "19": {
        "Não notei qualquer alteração no meu apetite.": 0,
        "Tenho um pouco menos de apetite que o habitual.": 1,
        "Tenho um pouco mais de apetite que o habitual.": 1,
        "O meu apetite é muito menor que o normal.": 2,
        "O meu apetite é muito maior que o normal.": 2,
        "Perdi por completo o apetite.": 3,
        "Anseio por comida o tempo todo.": 3,
    },
    "20": {
        "Concentro-me tão bem como antes.": 0,
        "Não me consigo concentrar tão bem como antes.": 1,
        "É difícil pensar em qualquer coisa por muito tempo.": 2,
        "Acho que não me consigo concentrar em nada.": 3,
    },
    "21": {
        "Não me sinto mais cansado que o habitual.": 0,
        "Canso-me mais facilmente que o costume.": 1,
        "Estou demasiado cansado para fazer uma série de coisas que costumava fazer.": 2,
        "Estou demasiado cansado para fazer a maior parte das coisas que costumava fazer.": 3,
    },
    "22": {
        "Não notei qualquer alteração no meu interesse sexual.": 1,
        "Sinto-me menos interessado sexualmente que o habitual.": 2,
        "Sinto-me muito menos interessado pela vida sexual.": 3,
        "Perdi por completo o interesse que tinha pela vida sexual.": 4,
    },
}
#
# Apply scoring to all 21 columns
# Apply scoring to each column based on its dictionary
for col in scoring.keys():
    if col in df.columns:  # Ensure the column exists
        df[col] = df[col].map(scoring[col])

# Compute total score per row (sum across all 21 columns)
df["Total Score"] = df[scoring.keys()].sum(axis=1)

# Reorder columns: Keep the first column, then insert "Total Score", then the rest
cols = list(df.columns)
df = df[[cols[0]] + ["Total Score"] + cols[1:-1]]

print(df)

# Save results
df.to_excel("G:\O meu disco\PhD\Study3\Results\scoreBDI.xlsx", index=False)

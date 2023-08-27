import pandas as pd


def main():
    df = pd.read_csv("predictions.csv")
    vanilla_correct = df.loc[df.targets == df.predicts_vanilla]
    perceiver_correct = df.loc[df.targets == df.predicts_perceiver]
    print(f"Vanilla correct: {len(vanilla_correct)} out of {len(df)}: {len(vanilla_correct)/len(df)*100}%")
    print(f"Perceiver correct: {len(perceiver_correct)} out of {len(df)}: {len(perceiver_correct)/len(df)*100}%")

    correct = (
        (vanilla_correct.targets.value_counts() / df.targets.value_counts())
        .dropna()
        .to_frame(name="score")
        .merge(df.targets.value_counts(), left_index=True, right_index=True)
        .reset_index()
    )
    print(correct.to_latex())


if __name__ == "__main__":
    main()

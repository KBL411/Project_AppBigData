import trainGradientBoosting
import trainRandomForest
import trainXGBoost

if __name__ == "__main__":
    try:
        print("Which model do you want to train ?\n[1] Gradient Boosting\n[2] Random Forest\n[3] XGBoost")
        choice = int(input())
        print("Which random state would you like to use ?")
        random_state = int(input())
        if choice != 3:
            print("How much n estimators would you like to use ?")
            n_estimators = int(input())
            if choice == 1:
                trainGradientBoosting.train(random_state, n_estimators)
            if choice == 2:
                trainRandomForest.train(random_state, n_estimators)
            if choice not in [1, 2]:
                raise ValueError
        else:
            trainXGBoost.train(random_state)
    except ValueError:
        print("Please input valid numbers only")



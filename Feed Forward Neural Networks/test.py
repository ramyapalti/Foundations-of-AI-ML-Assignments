import pickle as pkl
import argparse
from train import *

parser = argparse.ArgumentParser(prog = 'test', description = 'FFNN')
parser.add_argument('--num', dest = 'num', type = str, default = '100', help = 'number of test samples to predict')

def main():
    args = parser.parse_args()
    num = int(args.num)

    print("Predicting...")
    with open(f"./data/mnist_test.pkl", "rb") as file:
        test_data = pkl.load(file)
        print(f"test_x -- ({num}, 784); test_y -- ({num},1)")
        cnt = 0
        sample = num
        for i in range(sample):
            x_test = test_data[0][i].reshape(784, 1).T
            result = predict(x_test, test_data[1][i])
            if result[0] == test_data[1][i]:
                cnt+=1
            print("Prediction:", result[0], ", Actual:", test_data[1][i], ", Error:", result[1])

        print("Test Accuracy:", round(((1.0*cnt)/sample)*100.0, 4), "%")

   

    

if __name__ == "__main__":
    main()
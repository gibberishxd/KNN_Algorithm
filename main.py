import csv
import math


def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            processed_row = [float(val) if val.replace('.', '', 1).isdigit() else val for val in row]
            data.append(processed_row)
    return data


def calculate_euclidean_distance(vector1, vector2):
    squared_distance = sum((a - b) ** 2 for a, b in zip(vector1[:-1], vector2[:-1]))
    return squared_distance


def get_k_nearest_neighbors(train_data, test_instance, k):
    distances = [(calculate_euclidean_distance(test_instance, train_instance), train_instance[-1]) for train_instance in
                 train_data]
    distances.sort()
    return [neighbor[-1] for neighbor in distances[:k]]


def predict_class(neighbors):
    vote_counts = {}
    for neighbor in neighbors:
        vote_counts[neighbor] = vote_counts.get(neighbor, 0) + 1
    return max(vote_counts, key=vote_counts.get)


def evaluate_accuracy(test_data, predictions):
    correct_count = sum(1 for actual, predicted in zip(test_data, predictions) if actual[-1] == predicted)
    total_count = len(test_data)
    accuracy = (correct_count / total_count) * 100
    return accuracy


def classify_all_instances(train_data, test_data, k):
    predictions = [predict_class(get_k_nearest_neighbors(train_data, instance, k)) for instance in test_data]
    accuracy = evaluate_accuracy(test_data, predictions)
    print("Accuracy:", accuracy)
    print("\nTest Results:")
    for i, test_instance in enumerate(test_data):
        prediction = predictions[i]
        correctness = "Correct" if prediction == test_instance[-1] else "Incorrect"
        print("Row:", test_instance[:-1], "Actual Label:", test_instance[-1], "Predicted Label:", prediction,
              correctness)


def classify_user_input_instance(train_data, k):
    user_input = input("Enter observation attributes separated by commas: ").strip().split(',')
    user_input = [float(val) if val.replace('.', '', 1).isdigit() else val for val in user_input]
    prediction = predict_class(get_k_nearest_neighbors(train_data, user_input, k))
    print("Predicted class:", prediction)


def main():
    train_data = load_data("train.txt")
    test_data = load_data("test.txt")

    k = int(input("Enter the value of k: "))

    while True:
        print("\nOptions:")
        print("a) Classify all observations from the test set")
        print("b) Classify an observation provided by the user")
        print("c) Change k")
        print("d) Exit")

        choice = input("Choose an option: ").strip().lower()

        if choice == 'a':
            classify_all_instances(train_data, test_data, k)

        elif choice == 'b':
            classify_user_input_instance(train_data, k)

        elif choice == 'c':
            k = int(input("Enter the new value of k: "))

        elif choice == 'd':
            print("Exiting the program.")
            break

        else:
            print("Invalid option. Please choose a valid option.")


if __name__ == "__main__":
    main()

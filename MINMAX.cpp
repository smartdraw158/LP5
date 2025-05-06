#include <iostream>
#include <omp.h>
#include <climits>
#include <algorithm> 
using namespace std;

void min_reduction(int arr[], int n) {
    int min_value = INT_MAX;
    #pragma omp parallel for reduction(min:min_value)
    for (int i = 0; i < n; i++) {
        min_value = min(min_value, arr[i]);
    }
    cout << "Minimum value: " << min_value << endl;
}

void max_reduction(int arr[], int n) {
    int max_value = INT_MIN;
    #pragma omp parallel for reduction(max:max_value)
    for (int i = 0; i < n; i++) {
        max_value = max(max_value, arr[i]);
    }
    cout << "Maximum value: " << max_value << endl;
}

void sum_reduction(int arr[], int n) {
    int sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    cout << "Sum: " << sum << endl;
}

void average_reduction(int arr[], int n) {
    int sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    cout << "Average: " << static_cast<double>(sum) / n << endl;
}

int main() {
    int n;
    cout << "Enter total number of elements: ";
    cin >> n;

    if (n <= 0) {
        cout << "Invalid array size." << endl;
        return 1;
    }

    int* arr = new int[n];
    cout << "Enter elements:\n";
    for (int i = 0; i < n; i++) {
        cin >> arr[i];
    }

    // Perform parallel reductions
    min_reduction(arr, n);
    max_reduction(arr, n);
    sum_reduction(arr, n);
    average_reduction(arr, n);

    delete[] arr;
    return 0;
}


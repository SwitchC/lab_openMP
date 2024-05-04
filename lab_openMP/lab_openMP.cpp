#include <iostream>
#include <omp.h>
#include <vector>
#include <string>
#include <chrono>
using namespace std;
void print_matrix(vector<vector<int>> matrix) {
    for (int i = 0; i < matrix.size(); ++i) {
        for (int j = 0; j < matrix[0].size(); ++j) {
            cout << to_string(matrix[i][j]) + " ";
        }
        cout << '\n';
    }
    cout << '\n';
}
vector<vector<int>> initialize_matrix(int sizei, int sizej) {
    vector<vector<int>> matrix(sizei, vector<int>(sizej));
    for (int i = 0; i < sizei; ++i) {
        for (int j = 0; j < sizej; ++j) {
            matrix[i][j] = rand()%10;
        }
    }
    return matrix;
}
vector<vector<int>> split(vector<vector<int>> matrix,int quarterRow,int quarterCol) {
    if (matrix.size() == 1) {
        return matrix;
    }
    int newRows = matrix.size() / 2;
    int newCols = matrix[0].size() / 2;

    int startRow = (quarterRow - 1) * newRows;
    int startCol = (quarterCol - 1) * newCols;

    vector<vector<int>> result(newRows, vector<int>(newCols));
    for (int i = 0; i < newRows; i++) {
        for (int j = 0; j < newCols; j++) {
            result[i][j] = matrix[startRow + i][startCol + j];
        }
    }
    return result;
}
void matrix_multiply(vector<vector<int>>& C, vector<vector<int>>& A, vector<vector<int>>& B) {

    int n = A[0].size();
    if (n == 1) {
        C[0][0] = A[0][0] * B[0][0];
    }
    else {
        vector<vector<int>> T(n, vector<int>(n));

        vector<vector<int>> A11 = split(A, 1, 1);
        vector<vector<int>> A12 = split(A, 1, 2);
        vector<vector<int>> A21 = split(A, 2, 1);
        vector<vector<int>> A22 = split(A, 2, 2);

        vector<vector<int>> B11 = split(B, 1, 1);
        vector<vector<int>> B12 = split(B, 1, 2);
        vector<vector<int>> B21 = split(B, 2, 1);
        vector<vector<int>> B22 = split(B, 2, 2);

        vector<vector<int>> C11 = split(C, 1, 1);
        vector<vector<int>> C12 = split(C, 1, 2);
        vector<vector<int>> C21 = split(C, 2, 1);
        vector<vector<int>> C22 = split(C, 2, 2);

        vector<vector<int>> T11 = split(T, 1, 1);
        vector<vector<int>> T12 = split(T, 1, 2);
        vector<vector<int>> T21 = split(T, 2, 1);
        vector<vector<int>> T22 = split(T, 2, 2);

        matrix_multiply(C11, A11, B11);
        matrix_multiply(C12, A11, B12);
        matrix_multiply(C21, A21, B11);
        matrix_multiply(C22, A21, B12);
        matrix_multiply(C11, A11, B11);
        matrix_multiply(T11, A12, B21);
        matrix_multiply(T12, A12, B22);
        matrix_multiply(T21, A22, B21);
        matrix_multiply(T22, A22, B22);

        for (int i = 0; i < n / 2; i++) {
            for (int j = 0; j < n / 2; j++) {
                C[i][j] = C11[i][j] + T11[i][j];
            }
        }

        for (int i = n / 2; i < n; i++) {
            for (int j = 0; j < n / 2; j++) {
                C[i][j] = C21[i - n / 2][j] + T21[i - n / 2][j];
            }
        }
        for (int i = n / 2; i < n; i++) {
            for (int j = n / 2; j < n; j++) {
                C[i][j] = C22[i - n / 2][j - n / 2] + T22[i - n / 2][j - n / 2];
            }
        }

        for (int i = 0; i < n / 2; i++) {
            for (int j = n / 2; j < n; j++) {
                C[i][j] = C12[i][j - n / 2] + T12[i][j - n / 2];
            }
        }
    }
}
void p_matrix_multiply(vector<vector<int>>& C,vector<vector<int>>& A,vector<vector<int>>& B) {
    
    int n = A[0].size();
    if (n == 1) {
        C[0][0] = A[0][0] * B[0][0];
    }
    else {
        vector<vector<int>> T(n, vector<int>(n));

        vector<vector<int>> A11 = split(A, 1, 1);
        vector<vector<int>> A12 = split(A, 1, 2);
        vector<vector<int>> A21 = split(A, 2, 1);
        vector<vector<int>> A22 = split(A, 2, 2);

        vector<vector<int>> B11 = split(B, 1, 1);
        vector<vector<int>> B12 = split(B, 1, 2);
        vector<vector<int>> B21 = split(B, 2, 1);
        vector<vector<int>> B22 = split(B, 2, 2);

        vector<vector<int>> C11 = split(C, 1, 1);
        vector<vector<int>> C12 = split(C, 1, 2);
        vector<vector<int>> C21 = split(C, 2, 1);
        vector<vector<int>> C22 = split(C, 2, 2);

        vector<vector<int>> T11 = split(T, 1, 1);
        vector<vector<int>> T12 = split(T, 1, 2);
        vector<vector<int>> T21 = split(T, 2, 1);
        vector<vector<int>> T22 = split(T, 2, 2);
#pragma omp parallel shared(C11, C12, C22, C21, A11, A12, A21, A22, B11, B12, B21, B22, T11, T12, T21, T22)
        {
            #pragma omp single nowait 
            {
                p_matrix_multiply(C11, A11, B11);
            }
            #pragma omp single nowait 
            {
                p_matrix_multiply(C12, A11, B12);
            }
            #pragma omp single nowait 
            {
                p_matrix_multiply(C21, A21, B11);
            }
            #pragma omp single nowait 
            {
                p_matrix_multiply(C22, A21, B12);
            }
            #pragma omp single nowait
            {
                p_matrix_multiply(C11, A11, B11);
            }
            #pragma omp single nowait
            {
                p_matrix_multiply(T11, A12, B21);
            }
            #pragma omp single nowait
            {
                p_matrix_multiply(T12, A12, B22);
            }
            #pragma omp single nowait
            {
                p_matrix_multiply(T21, A22, B21);
            }
            #pragma omp single nowait
            {
                p_matrix_multiply(T22, A22, B22);
            }
            #pragma omp barrier
        }
        

        #pragma omp parallel for
        for (int i = 0; i < n/2; i++) {
            for (int j = 0; j < n/2; j++) {
                C[i][j] = C11[i][j] + T11[i][j];
            }
        }
        #pragma omp parallel for
        for (int i = n/2; i < n; i++) {
            for (int j = 0; j < n / 2; j++) {
                C[i][j] = C21[i-n/2][j] + T21[i-n/2][j];
            }
        }
        #pragma omp parallel for
        for (int i = n / 2; i < n; i++) {
            for (int j = n/2; j < n; j++) {
                C[i][j] = C22[i - n / 2][j-n/2] + T22[i - n / 2][j-n/2];
            }
        }
        #pragma omp parallel for
        for (int i =0; i < n/2; i++) {
            for (int j = n / 2; j < n; j++) {
                C[i][j] = C12[i][j - n / 2] + T12[i][j - n / 2];
            }
        }
    }
}
int main() {
    auto A = initialize_matrix(64, 64);
    auto B = initialize_matrix(64, 64);
    /*print_matrix(A);
    print_matrix(B);*/
    vector<vector<int>> C(A.size(), vector<int>(A.size()));
    /*p_matrix_multiply(C, A, B);
    print_matrix(C);*/
    chrono::steady_clock::time_point begin = chrono::steady_clock::now();
    p_matrix_multiply(C, A, B);
    chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    cout << "parallelism: " + to_string(std::chrono::duration_cast<std::chrono::seconds>(end - begin).count()) + " seconds"<<endl;
    begin = chrono::steady_clock::now();
    matrix_multiply(C, A, B);
    end = std::chrono::steady_clock::now();
    cout << "No parallelism: " + to_string(std::chrono::duration_cast<std::chrono::seconds>(end - begin).count()) + " seconds" << endl;

}
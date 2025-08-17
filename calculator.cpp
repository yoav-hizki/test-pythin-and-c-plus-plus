#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>

/**
 * Simple Calculator - Basic mathematical operations
 * Demonstrates C++ features including:
 * - Classes and methods
 * - STL containers
 * - Mathematical functions
 * - Input/output operations
 */

class Calculator {
private:
    std::vector<double> history;

public:
    // Basic arithmetic operations
    double add(double a, double b) {
        double result = a + b;
        history.push_back(result);
        return result;
    }

    double subtract(double a, double b) {
        double result = a - b;
        history.push_back(result);
        return result;
    }

    double multiply(double a, double b) {
        double result = a * b;
        history.push_back(result);
        return result;
    }

    double divide(double a, double b) {
        if (b == 0) {
            std::cout << "Error: Division by zero!" << std::endl;
            return 0;
        }
        double result = a / b;
        history.push_back(result);
        return result;
    }

    // Advanced operations
    double power(double base, double exponent) {
        double result = std::pow(base, exponent);
        history.push_back(result);
        return result;
    }

    double squareRoot(double value) {
        if (value < 0) {
            std::cout << "Error: Cannot calculate square root of negative number!" << std::endl;
            return 0;
        }
        double result = std::sqrt(value);
        history.push_back(result);
        return result;
    }

    // Statistical functions
    double calculateMean(const std::vector<double>& numbers) {
        if (numbers.empty()) return 0;

        double sum = 0;
        for (double num : numbers) {
            sum += num;
        }
        return sum / numbers.size();
    }

    double findMax(const std::vector<double>& numbers) {
        if (numbers.empty()) return 0;
        return *std::max_element(numbers.begin(), numbers.end());
    }

    double findMin(const std::vector<double>& numbers) {
        if (numbers.empty()) return 0;
        return *std::min_element(numbers.begin(), numbers.end());
    }

    // History management
    void printHistory() {
        std::cout << "Calculation History:" << std::endl;
        for (size_t i = 0; i < history.size(); ++i) {
            std::cout << i + 1 << ": " << std::fixed << std::setprecision(2)
                      << history[i] << std::endl;
        }
    }

    void clearHistory() {
        history.clear();
        std::cout << "History cleared." << std::endl;
    }

    size_t getHistorySize() const {
        return history.size();
    }
};

void displayMenu() {
    std::cout << "\n=== CALCULATOR MENU ===" << std::endl;
    std::cout << "1. Addition" << std::endl;
    std::cout << "2. Subtraction" << std::endl;
    std::cout << "3. Multiplication" << std::endl;
    std::cout << "4. Division" << std::endl;
    std::cout << "5. Power" << std::endl;
    std::cout << "6. Square Root" << std::endl;
    std::cout << "7. Calculate Mean" << std::endl;
    std::cout << "8. Find Max/Min" << std::endl;
    std::cout << "9. Show History" << std::endl;
    std::cout << "10. Clear History" << std::endl;
    std::cout << "0. Exit" << std::endl;
    std::cout << "Choice: ";
}

int main() {
    Calculator calc;
    int choice;
    double a, b, result;

    std::cout << "Welcome to the C++ Calculator!" << std::endl;

    do {
        displayMenu();
        std::cin >> choice;

        switch (choice) {
            case 1:
                std::cout << "Enter two numbers: ";
                std::cin >> a >> b;
                result = calc.add(a, b);
                std::cout << "Result: " << result << std::endl;
                break;

            case 2:
                std::cout << "Enter two numbers: ";
                std::cin >> a >> b;
                result = calc.subtract(a, b);
                std::cout << "Result: " << result << std::endl;
                break;

            case 3:
                std::cout << "Enter two numbers: ";
                std::cin >> a >> b;
                result = calc.multiply(a, b);
                std::cout << "Result: " << result << std::endl;
                break;

            case 4:
                std::cout << "Enter two numbers: ";
                std::cin >> a >> b;
                result = calc.divide(a, b);
                std::cout << "Result: " << result << std::endl;
                break;

            case 5:
                std::cout << "Enter base and exponent: ";
                std::cin >> a >> b;
                result = calc.power(a, b);
                std::cout << "Result: " << result << std::endl;
                break;

            case 6:
                std::cout << "Enter number: ";
                std::cin >> a;
                result = calc.squareRoot(a);
                std::cout << "Result: " << result << std::endl;
                break;

            case 7: {
                int count;
                std::cout << "How many numbers? ";
                std::cin >> count;
                std::vector<double> numbers(count);
                std::cout << "Enter " << count << " numbers: ";
                for (int i = 0; i < count; ++i) {
                    std::cin >> numbers[i];
                }
                result = calc.calculateMean(numbers);
                std::cout << "Mean: " << result << std::endl;
                break;
            }

            case 8: {
                int count;
                std::cout << "How many numbers? ";
                std::cin >> count;
                std::vector<double> numbers(count);
                std::cout << "Enter " << count << " numbers: ";
                for (int i = 0; i < count; ++i) {
                    std::cin >> numbers[i];
                }
                std::cout << "Max: " << calc.findMax(numbers) << std::endl;
                std::cout << "Min: " << calc.findMin(numbers) << std::endl;
                break;
            }

            case 9:
                calc.printHistory();
                break;

            case 10:
                calc.clearHistory();
                break;

            case 0:
                std::cout << "Thank you for using the calculator!" << std::endl;
                break;

            default:
                std::cout << "Invalid choice. Please try again." << std::endl;
        }

    } while (choice != 0);

    return 0;
}

from random import *
'''
---句子长度列表: [23, 23, 83, 83, 83, 28, 53, 16, 18, 74, 17, 32, 32, 56, 36, 36, 36, 18, 41, 71, 21, 53, 53, 20, 30, 30, 30, 21, 21, 11, 33, 33, 15, 62, 62, 62, 62, 19, 30, 55, 41, 41, 48, 48, 16, 42, 42, 42, 42, 16, 28, 49, 19, 19, 38, 90, 90, 28, 86, 96, 26, 26, 22, 22, 46, 46, 24, 28, 24, 54, 23, 71, 71, 71, 71, 71, 71, 17, 18, 18, 34, 34, 34, 34, 16, 35, 25, 68, 36, 36, 58, 29, 19, 59, 59, 59, 59, 19, 41, 24, 24, 50, 50, 50, 50, 50, 50, 50, 50, 50, 35, 35, 40, 40, 24, 24, 38, 38, 40, 59, 59, 59, 75, 75, 75, 42, 24, 42, 44, 27, 21, 67, 67, 97, 67, 67, 30, 56, 56, 56, 13, 20, 37, 48, 48, 72, 35, 35, 26, 26, 23, 52, 52, 52, 23, 23, 56, 56, 37, 37, 73, 24, 36, 36, 36, 21, 18, 35, 36, 24, 24, 28, 27, 45, 45, 53, 53, 89, 89, 44, 22, 64, 17, 178, 178, 178, 178, 21, 54, 33, 42, 46, 27, 27, 65, 51, 134, 134, 27, 22, 33, 22, 25, 46, 46, 48, 48, 69, 22, 15, 28, 28, 29, 50, 76, 34, 24, 24, 18, 37, 98, 98, 30, 91, 91, 30, 91, 91, 44, 14, 31, 73, 29, 29, 29, 63, 40, 40, 19, 19, 19, 19, 56, 71, 107, 56, 56, 33, 73, 73, 102, 94, 94, 38, 9, 34, 34, 62, 62, 28, 22, 22, 22, 49, 49, 38, 38, 38, 33, 33, 75, 75, 92, 22, 50, 50, 17, 45, 18, 41, 53, 41, 53, 31, 38, 34, 34, 34, 18, 92, 35, 35, 35, 34, 34, 22, 22, 38, 73, 73, 24, 24, 32, 22, 22, 16, 16, 12, 33, 52, 60, 20, 20, 10, 68, 15, 15, 30, 130, 31, 31, 43, 43, 69, 69, 69, 100, 100, 100, 100, 100, 100, 100, 100, 18, 19, 22, 57, 57, 57, 35, 35, 68, 14, 27, 27, 28, 28, 45, 45, 17, 48, 27, 27, 10, 61, 61, 61, 84, 84, 43, 24, 24, 18, 18, 23, 37, 37, 37, 37, 26, 26, 23, 23, 36, 22, 98, 76, 64, 53, 53, 75, 13, 26, 26, 31, 31, 80, 89, 30, 30, 26, 25, 25, 25, 11, 17, 17, 13, 16, 19, 19, 25, 18, 29, 29, 84, 134, 116, 84, 20, 35, 38, 38, 38, 59, 59, 59, 59, 79, 102, 59, 79, 102, 15, 30, 9, 31, 31, 25, 25, 25, 50, 30, 30, 30, 62, 35, 26, 26, 26, 26, 26, 38, 51, 23, 23, 18, 18, 43, 48, 48, 48, 48, 29, 63, 73, 63, 73, 63, 73, 63, 73, 62, 51, 51, 40, 26, 26, 26, 36, 36, 36, 36, 49, 49, 18, 18, 27, 27, 27, 34, 29, 29, 39, 52, 29, 58, 49, 49, 34, 34, 34, 45, 103, 103, 103, 18, 30, 18, 32, 47, 32, 47, 58, 58, 58, 43, 43, 25, 25, 25, 29, 38, 29, 18, 18, 38, 38, 33, 33, 16, 24, 18, 18, 39, 28, 15, 24, 24, 24, 74, 74, 106, 117, 114, 74, 74, 106, 41, 41, 82, 41, 41, 41, 67, 23, 45, 58, 87, 87, 19, 23, 23, 34, 41, 41, 53, 53, 53, 53, 22, 32, 57, 39, 39, 17, 29, 22, 30, 22, 17, 17, 41, 22, 49, 15, 13, 35, 18, 41, 52, 52, 43, 43, 43, 43, 20, 22, 22, 46, 46, 39, 56, 56, 20, 43, 43, 10, 24, 34, 15, 43, 14, 18, 18, 33, 19, 19, 19, 85, 85, 57, 57, 57, 57, 20, 13, 57, 57, 57, 29, 29, 29, 18, 18, 60, 31, 20, 51, 51, 51, 51, 20, 20, 31, 31, 31, 15, 27, 27, 15, 42, 42, 52, 52, 21, 21, 21, 30, 30, 30, 21, 21, 14, 14, 17, 42, 42, 42, 34, 34, 34, 54, 54, 54, 29, 29, 29, 19, 107, 107, 107, 107, 107, 107, 107, 45, 45, 18, 18, 17, 17, 32, 32, 34, 34, 34, 34, 17, 28, 17, 17, 19, 19, 55, 55, 42, 42, 42, 42, 59, 29, 29, 21, 21, 21, 21, 58, 58, 58, 25, 39, 51, 51, 15, 77, 77, 19, 33, 175, 175, 175, 175, 175, 24, 24, 18, 30, 27, 26, 26, 26, 19, 20, 19, 19, 55, 61, 43, 12, 12, 19, 41, 41, 22, 61, 20, 50, 18, 30, 28, 16, 16, 47, 47, 47, 47, 47, 47, 47, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 24, 24, 24, 11, 15, 14, 31, 31, 48, 48, 15, 15, 10, 22, 17, 17, 27, 27, 27, 27, 64, 64, 37, 40, 40, 40, 40, 40, 40, 13, 53, 53, 53, 58, 58, 58, 58, 58, 58, 58, 58, 18, 26, 17, 44, 44, 22, 17, 41, 17, 41, 28, 28, 53, 53, 80, 80, 20, 20, 33, 28, 57, 57, 18, 57, 57, 49, 49, 49, 14, 14, 31, 31, 25, 30, 30, 30, 31, 31, 87, 87, 26, 26, 30, 30, 30, 16, 27, 13, 38, 38, 38, 19, 51, 19, 24, 58, 76, 24, 24, 24, 24, 24, 18, 18, 18, 31, 31, 31, 31, 22, 57, 30, 26, 26, 53, 24, 24, 36, 24, 8, 70, 80, 70, 80, 70, 80, 70, 80, 54, 54, 33, 46, 39, 39, 39, 79, 39, 79, 87, 43, 43, 33, 33, 33, 33, 20, 10, 31, 32, 18, 33, 48, 33, 48, 58, 43, 43, 43, 25, 25, 26, 26, 26, 26, 33, 69, 79, 31, 31, 59, 59, 31, 28, 14, 14, 41, 23, 23, 23, 46, 46, 46, 54, 27, 43, 43, 30, 30, 30, 30, 36, 13, 95, 117, 108, 73, 48, 22, 19, 19, 19, 26, 21, 18, 42, 81, 81, 120, 42, 81, 120, 120, 23, 23, 23, 57, 22, 21, 21, 20, 83, 83, 83, 35, 20, 22, 42, 22, 12, 12, 26, 26, 58, 75, 102, 102, 25, 22, 17, 25, 25, 11, 20, 33, 33, 30, 30, 48, 48, 48, 48, 22, 18, 26, 55, 55, 55, 55, 55, 55, 28, 28, 28, 52, 52, 52, 52, 52, 52, 52, 52, 52, 33, 33, 87, 87, 87, 87, 18, 25, 25, 25, 26, 26, 21, 21, 19, 19, 24, 24, 19, 15, 12, 28, 28, 28, 54, 54, 54, 86, 32, 25, 32, 54, 54, 20, 46, 8, 23, 58, 58, 58, 43, 43, 43, 23, 23, 33, 59, 59, 33, 59, 59, 56, 56, 56, 56, 56, 56, 56, 27, 27, 27, 51, 51, 62, 24, 38, 38, 38, 38, 38, 46, 46, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 16, 46, 46, 64, 64, 46, 46, 64, 64, 33, 91, 49, 16, 41, 41, 50, 50, 50, 50, 50, 37, 15, 65, 101, 17, 53, 65, 53, 26, 26, 11, 123, 123, 123, 74, 13, 47, 47, 47, 47, 16, 79, 79, 79, 79, 79, 79, 79, 79, 79, 17, 32, 32, 9, 32, 32, 32, 32, 38, 58, 58, 58, 58, 58, 24, 24, 51, 51, 31, 31, 31, 18, 29, 73, 73, 43, 43, 31, 68, 82, 104, 29, 29, 19, 19, 19, 14, 54, 54, 54, 78, 47, 63, 40, 40, 40, 21, 39, 39, 17, 17, 32, 32, 32, 16, 58, 58, 65, 21, 21, 21, 42, 42, 42, 19, 40, 40, 56, 76, 64, 64, 64, 64, 32, 32, 30, 30, 30, 30, 18, 18, 33, 32, 15, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 16, 50, 50, 68, 68, 68, 21, 21, 21, 18, 47, 47, 47, 47, 17, 17, 77, 77, 77, 77, 77, 77, 48, 48, 48, 48, 66, 66, 23, 24, 14, 30, 30, 30, 30, 30, 12, 41, 17, 46, 24, 13, 23, 23, 58, 58, 48, 48, 48, 48, 48, 48, 95, 95, 24, 28, 28, 28, 18, 39, 58, 58, 58, 22, 83, 48, 48, 69, 69, 62, 62, 62, 62, 62, 62, 21, 20, 43, 43, 43, 20, 19, 29, 20, 29, 33, 33, 93, 93, 93, 93, 93, 56, 56, 56, 27, 43, 51, 54, 35, 65, 30, 23, 28, 51, 16, 39, 39, 115, 76, 19, 36, 17, 17, 9, 14, 22, 25, 25, 26, 26, 15, 29, 29, 56, 56, 56, 18, 56, 18, 18, 37, 37, 37, 37, 37, 22, 22, 22, 21, 21, 41, 29, 19, 30, 30, 27, 16, 16, 64, 48, 48, 77, 77, 23, 60, 60, 19, 19, 19, 35, 33, 41, 41, 41, 30, 30, 51, 51, 65, 81, 56, 37, 37, 62, 62, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 9, 21, 37, 60, 51, 44, 28, 13, 34, 20, 20, 20, 20, 11, 11, 33, 33, 33, 64, 64, 17, 17, 27, 27, 40, 13, 22, 9, 25, 25, 25, 33, 57, 57, 57, 57, 57, 117, 117, 117, 19, 18, 15, 28, 31, 38, 23, 35, 51, 14, 26, 42, 33, 78, 53, 70, 14, 19, 19, 22, 22, 38, 28, 24, 67, 104, 52, 36, 68, 68, 68, 68, 68, 68, 68, 66, 103, 66, 66, 41, 41, 37, 24, 24, 24, 24, 19, 19, 47, 47, 64, 64, 16, 16, 56, 56, 56, 56, 28, 28, 18, 18, 45, 45, 45, 45, 45, 46, 49, 36, 36, 36, 141, 141, 141, 141, 167, 184, 94, 94, 94, 94, 94, 113, 62, 62, 62, 62, 30, 30, 30, 30, 34, 34, 34, 91, 91, 91, 91, 91, 31, 31, 45, 45, 45, 19, 36, 48, 48, 29, 51, 29, 51, 22, 124, 124, 124, 124, 32, 39, 39, 39, 95, 22, 22, 49, 49, 52, 23, 20, 77, 77, 50, 50, 24, 35, 25, 48, 12, 29, 29, 29, 37, 37, 37, 47, 33, 33, 18, 18, 53, 53, 53, 53, 53, 22, 34, 17, 17, 26, 26, 34, 34, 36, 22, 22, 25, 25, 44, 54, 12, 23, 27, 17, 23, 17, 47, 47, 47, 59, 59, 59, 59, 59, 35, 35, 20, 24, 24, 38, 38, 38, 21, 21, 23, 23, 23, 23, 23, 14, 14, 14, 35, 35, 35, 20, 20, 20, 39, 39, 18, 18, 21, 21, 28, 28, 16, 16, 28, 28, 14, 14, 63, 63, 63, 63, 63, 63, 30, 30, 30, 52, 52, 52, 34, 34, 21, 21, 21, 33, 33, 33, 72, 25, 25, 25, 34, 34, 34, 23, 23, 23, 38, 38, 38, 22, 22, 41, 41, 41, 41, 52, 69, 69, 69, 69, 69, 69, 69, 69, 22, 23, 23, 32, 32, 32, 11, 11, 11, 32, 32, 26, 26, 26, 26, 26, 33, 33, 33, 53, 53, 53, 72, 27, 27, 27, 27, 31, 31, 31, 37, 37, 37, 23, 23, 23, 23, 54, 83, 63, 63, 50, 32, 46, 59, 75, 16, 16, 30, 30, 30, 30, 30, 38, 38, 38, 38, 38, 38, 49, 49, 18, 18, 31, 31, 31, 31, 31, 38, 31, 31, 31, 31, 31, 47, 47, 20, 20, 40, 40, 40, 40, 40, 40, 60, 60, 60, 17, 18, 18, 18, 40, 40, 31, 31, 60, 60, 18, 35, 35, 35, 35, 29, 57, 68, 57, 57, 76, 76, 10, 44, 34, 23, 23, 20, 20, 20, 48, 48, 94, 94, 48, 48, 46, 25, 36, 45, 36, 36, 36, 24, 24, 20, 20, 20, 20, 41, 17, 17, 15, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 36, 36, 39, 39, 39, 82, 43, 21, 21, 82, 86, 86, 129, 129, 132, 86, 86, 86, 86, 129, 58, 58, 58, 82, 82, 46, 46, 24, 42, 42, 15, 29, 14, 44, 30, 30, 11, 34, 26, 26, 17, 38, 17, 17, 38, 23, 23, 26, 26, 26, 26, 35, 86, 35, 35, 86, 71, 71, 71, 71, 20, 42, 80, 40, 40, 40, 40, 40, 40, 33, 38, 38, 38, 38, 38, 38, 33, 33, 25, 25, 25, 23, 23, 52, 61, 23, 23, 23, 52, 61, 16, 55, 65, 16, 37, 22, 22, 22, 22, 59, 59, 59, 20, 20, 32, 45, 45, 23, 19, 62, 62, 62, 62, 62, 89, 89, 89, 89, 89, 19, 59, 19, 49, 49, 49, 40, 40, 40, 18, 18, 48, 21, 21, 21, 30, 30, 65, 19, 19, 65, 37, 28, 13, 13, 32, 13, 35, 47, 35, 13, 35, 47, 35, 13, 13, 28, 19, 44, 19, 19, 17, 24, 24, 24, 36, 48, 62, 80, 14, 18, 48, 24, 24, 11, 39, 28, 28, 28, 28, 28, 21, 41, 21, 20, 32, 42, 42, 42, 16, 19, 24, 18, 15, 16, 42, 42, 30, 30, 30, 30, 27, 27, 12, 24, 78, 78, 78, 78, 78, 78, 24, 30, 30, 22, 22, 17, 36, 36, 36, 36, 36, 36, 36, 36, 29, 29, 55, 29, 29, 55, 29, 29, 55, 26, 24, 24, 46, 46, 46, 46, 46, 46, 46, 22, 55, 55, 43, 43, 94, 94, 94, 94, 94, 94, 14, 43, 77, 20, 23, 61, 61, 61, 61, 61, 34, 34, 19, 12, 12, 23, 30, 44, 20, 25, 35, 20, 25, 35, 109, 109, 11, 17, 21, 22, 23, 23, 56, 20, 36, 31, 35, 35, 39, 39, 39, 54, 54, 23, 23, 41, 17, 31, 31, 43, 43, 59, 50, 50, 50, 50, 50, 50, 43, 43, 24, 30, 30, 26, 26, 18, 16, 53, 53, 53, 53, 17, 17, 45, 45, 45, 23, 23, 45, 45, 23, 54, 54, 33, 43, 43, 43, 37, 37, 37, 37, 26, 26, 26, 20, 35, 35, 35, 13, 19, 19, 50, 23, 23, 53, 53, 40, 40, 40, 40, 40, 40, 40, 56, 56, 56, 56, 25, 25, 24, 21, 32, 17, 29, 61, 61, 61, 29, 29, 32, 32, 32, 22, 22, 22, 22, 29, 58, 58, 23, 23, 16, 23, 23, 29, 19, 20, 40, 40, 40, 61, 58, 37, 80, 58, 37, 80, 22, 21, 19, 40, 40, 40, 40, 40, 40, 15, 49, 57, 57, 57, 57, 57, 46, 46, 46, 46, 33, 33, 33, 46, 30, 30, 24, 24, 24, 20, 32, 32, 46, 25, 15, 46, 26, 35, 26, 35, 16, 30, 30, 30, 21, 51, 51, 24, 15, 20, 20, 32, 32, 53, 71, 23, 43, 69, 69, 13, 35, 35, 24, 24, 39, 24, 36, 15, 57, 57, 26, 37, 57, 57, 36, 57, 37, 29, 29, 30, 30, 42, 42, 27, 27, 27, 21, 25, 48, 23, 37, 37, 19, 44, 28, 28, 27, 80, 80, 124, 207, 180, 207, 41, 19, 19, 46, 33, 57, 90, 90, 32, 52, 73, 21, 35, 50, 17, 26, 37, 20, 20, 35, 35, 13, 45, 45, 45, 45, 31, 22, 80, 50, 80, 81, 86, 81, 81, 86, 81, 86, 81, 86, 81, 81, 86, 81, 86, 60, 60, 60, 27, 101, 101, 101, 53, 101, 53, 65, 65, 65, 65, 65, 65, 65, 81, 81, 81, 48, 48, 48, 48, 23, 23, 41, 41, 20, 45, 45, 45, 20, 45, 20, 45, 45, 49, 49, 49, 49, 20, 86, 86, 86, 86, 63, 63, 63, 63, 63, 63, 63, 63, 28, 28, 28, 28, 85, 85, 26, 56, 56, 56, 56, 52, 52, 52, 52, 23, 39, 39, 39, 79, 79, 79, 47, 47, 22, 40, 40, 40, 105, 105, 105, 105, 105, 105, 105, 105, 105, 41, 41, 20, 52, 13, 58, 58, 45, 38, 38, 38, 38, 56, 77, 77, 104, 18, 36, 36, 36, 36, 21, 21, 21, 21, 21, 46, 46, 46, 15, 17, 47, 22, 64, 64, 29, 47, 66, 84, 53, 53, 66, 89, 53, 53, 53, 53, 53, 66, 66, 21, 26, 43, 89, 89, 53, 53, 52, 43, 26, 26, 61, 61, 30, 30, 30, 40, 33, 33, 33, 52, 26, 52, 107, 107, 107, 107, 107, 41, 56, 25, 39, 71, 71, 23, 19, 8, 18, 18, 23, 35, 28, 25, 39, 44, 44, 49, 116, 34, 34, 34, 34, 30, 30, 83, 30, 83, 25, 23, 41, 41, 41, 34, 68, 68, 13, 13, 34, 34, 34, 47, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 41, 41, 41, 45, 25, 33, 79, 33, 33, 79, 33, 79, 86, 86, 96, 119, 119, 119, 26, 26, 26, 26, 37, 37, 37, 30, 30, 20, 20, 40, 40, 20, 21, 21, 43, 43, 21, 21, 34, 34, 20, 20, 25, 24, 24, 37, 37, 37, 22, 22, 63, 63, 63, 63, 41, 41, 41, 41, 41, 37, 37, 21, 21, 25, 25, 123, 98, 98, 23, 23, 19, 19, 19, 25, 25, 25, 26, 26, 33, 33, 33, 33, 33, 33, 33, 33, 33, 21, 81, 21, 83, 60, 60, 83, 62, 20, 41, 21, 21, 19, 19, 55, 55, 55, 55, 55, 55, 40, 21, 21, 17, 57, 77, 33, 17, 23, 19, 19, 25, 53, 18, 18, 91, 53, 53, 53, 53, 53, 96, 96, 96, 96, 17, 17, 57, 77, 33, 28, 28, 28, 44, 71, 49, 38, 30, 30, 30, 30, 30, 28, 13, 58, 58, 58, 21, 16, 21, 16, 36, 36, 58, 58, 26, 43, 43, 67, 83, 43, 67, 83, 44, 44, 68, 85, 44, 68, 85, 31, 29, 29, 29, 9, 43, 43, 28, 28, 28, 44, 71, 49, 38, 30, 30, 30, 30, 30, 29, 13, 58, 58, 58, 58, 58, 58, 22, 22, 36, 36, 16, 16, 26, 26, 48, 20, 20, 20, 17, 17, 29, 29, 29, 56, 56, 10, 72, 72, 53, 53, 53, 33, 33, 30, 30, 30, 30, 28, 28, 28, 23, 28, 28, 28, 23, 62, 43, 16, 62, 43, 16, 16, 40, 32, 32, 32, 32, 16, 16, 25, 40, 50, 40, 17, 65, 65, 65, 113, 113, 113, 113, 113, 113, 22, 13, 61, 61, 61, 43, 43, 46, 46, 31, 31, 31, 49, 86, 86, 86, 37, 37, 37, 21, 38, 69, 69, 81, 58, 58, 31, 23, 58, 23, 14, 14, 14, 87, 68, 68, 18, 41, 23, 68, 23, 16, 25, 41, 25, 25, 41, 75, 75, 75, 75, 75, 75, 98, 23, 19, 45, 45, 45, 45, 19, 39, 39, 26, 26, 39, 39, 26, 64, 66, 66, 66, 66, 66, 47, 24, 24, 89, 89, 89, 89, 89, 47, 47, 47, 59, 36, 36, 61, 36, 31, 31, 31, 71, 71, 20, 20, 20, 27, 27, 27, 22, 22, 22, 28, 45, 45, 28, 28, 17, 18, 52, 19, 19, 30, 73, 30, 48, 48, 19, 19, 18, 18, 18, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 41, 41, 41, 41, 41, 15, 34, 29, 29, 29, 29, 35, 18, 96, 61, 29, 29, 29, 29, 29, 29, 13, 37, 11, 26, 72, 72, 72, 72, 36, 47, 57, 32, 68, 68, 18, 65, 90, 18, 10, 68, 112, 140, 140, 140, 18, 18, 26, 26, 34, 34, 34, 22, 32, 32, 32, 34, 34, 55, 55, 55, 55, 55, 55, 55, 12, 29, 29, 29, 33, 33, 33, 22, 47, 47, 47, 23, 29, 29, 17, 67, 50, 50, 50, 50, 50, 50, 74, 74, 74, 52, 54, 41, 28, 28, 17, 29, 29, 29, 27, 27, 21, 21, 18, 30, 30, 30, 15, 15, 41, 41, 41, 41, 18, 33, 33, 33, 26, 68, 68, 15, 41, 17, 17, 32, 16, 36, 36, 16, 36, 36, 65, 65, 32, 19, 19, 24, 24, 24, 24, 42, 17, 19, 19, 33, 33, 33, 20, 20, 28, 28, 28, 20, 20, 42, 30, 30, 34, 34, 34, 34, 34, 28, 80, 80, 80, 80, 52, 20, 22, 30, 30, 28, 38, 28, 38, 45, 17, 45, 45, 49, 49, 49, 20, 30, 60, 30, 30, 32, 32, 62, 17, 17, 24, 18, 58, 50, 24, 50, 50, 27, 21, 21, 21, 28, 18, 36, 36, 43, 43, 21, 21, 29, 37, 10, 32, 57, 57, 57, 20, 45, 57, 45, 57, 45, 57, 20, 23, 22, 22, 43, 15, 48, 19, 19, 34, 51, 58, 10, 34, 19, 19, 54, 35, 35, 35, 35, 35, 25, 25, 31, 25, 25, 38, 38, 26, 26, 17, 13, 28, 28, 44, 23, 18, 18, 40, 24, 22, 22, 22, 15, 15, 30, 30, 9, 18, 19, 37, 37, 37, 37, 37, 37, 33, 20, 20, 32, 32, 32, 18, 18, 38, 38, 38, 15, 29, 29, 32, 21, 32, 52, 52, 52, 19, 17, 13, 13, 24, 33, 33, 12, 45, 33, 19, 18, 40, 40, 15, 45, 45, 17, 70, 70, 70, 70, 53, 53, 17, 67, 67, 67, 50, 21, 52, 79, 97, 52, 79, 97, 28, 55, 55, 55, 12, 12, 28, 28, 28, 24, 24, 24, 24, 24, 33, 33, 33, 65, 32, 32, 24, 19, 19, 42, 42, 23, 23, 23, 38, 38, 38, 45, 45, 45, 26, 26, 43, 43, 43, 33, 33, 33, 28, 28, 28, 16, 42, 42, 13, 49, 49, 36, 82, 82, 27, 21, 17, 29, 29, 29, 29, 29, 29, 73, 73, 73, 54, 54, 48, 69, 48, 69, 69, 48, 69, 23, 19, 19, 32, 50, 102, 102, 102, 17, 23, 74, 74, 74, 74, 16, 22, 54, 54, 54, 38, 38, 38, 23, 38, 58, 58, 58, 58, 58, 67, 67, 67, 67, 67, 67, 74, 58, 74, 74, 74, 74, 74, 74, 74, 13, 65, 65, 65, 65, 32, 65, 65, 65, 65, 65, 65, 65, 65, 65, 48, 48, 48, 70, 70, 70, 70, 102, 102, 102, 51, 51, 35, 35, 35, 35, 35, 35, 81, 81, 43, 43, 47, 32, 47, 32, 32, 47, 32, 26, 26, 50, 50, 50, 50, 55, 55, 55, 55, 13, 66, 66, 66, 26, 83, 83, 83, 83, 83, 83, 54, 54, 54, 54, 29, 56, 56, 56, 56, 13, 18, 36, 36, 35, 35, 35, 35, 34, 34, 25, 10, 47, 47, 22, 95, 95, 22, 22, 23, 23, 17, 17, 17, 36, 36, 19, 19, 50, 29, 31, 24, 24, 17, 22, 22, 25, 25, 16, 54, 54, 54, 54, 21, 11, 49, 49, 49, 49, 26, 44, 67, 34, 34, 12, 12, 35, 48, 7, 20, 20, 16, 16, 22, 41, 21, 21, 21, 21, 16, 21, 21, 19, 19, 29, 32, 32, 32, 32, 32, 27, 27, 27, 43, 16, 26, 50, 26, 50, 24, 40, 40, 20, 16, 16, 32, 49, 16, 39, 39, 39, 39, 39, 17, 17, 33, 16, 24, 19, 28, 39, 39, 17, 18, 18, 24, 17, 64, 64, 64, 64, 64, 64, 8, 19, 36, 36, 24, 24, 24, 34, 42, 42, 42, 42, 42, 34, 34, 44, 44, 11, 16, 16, 28, 28, 28, 22, 87, 87, 65, 65, 65, 65, 19, 75, 75, 56, 56, 40, 40, 30, 30, 38, 52, 52, 52, 16, 59, 59, 24, 38, 38, 38, 33, 26, 26, 33, 45, 45, 69, 24, 24, 24, 53, 36, 36, 36, 40, 40, 40, 40, 55, 55, 55, 55, 55, 25, 37, 8, 44, 44, 32, 32, 43, 43, 43, 23, 55, 20, 28, 27, 55, 27, 23, 35, 33, 33, 19, 34, 44, 44, 19, 32, 15, 58, 58, 25, 37, 34, 62, 80, 80, 94, 25, 46, 71, 22, 41, 57, 26, 17, 37, 11, 16, 43, 22, 22, 63, 60, 60, 26, 26, 73, 73, 47, 47, 48, 48, 13, 49, 49, 36, 36, 83, 56, 21, 83, 21, 27, 78, 78, 78, 78, 78, 65, 65, 65, 65, 65, 65, 65, 65, 62, 49, 49, 109, 109, 109, 109, 59, 59, 33, 33, 36, 36, 36, 66, 66, 66, 43, 43, 43, 45, 45, 45, 45, 25, 25, 20, 44, 44, 44, 20, 20, 44, 44, 44, 57, 57, 17, 37, 37, 37, 37, 50, 50, 50, 97, 97, 97, 47, 47, 23, 16, 16, 62, 49, 90, 90, 26, 54, 54, 54, 54, 65, 49, 49, 49, 49, 38, 38, 38, 38, 40, 40, 40, 40, 19, 19, 27, 25, 25, 25, 62, 53, 53, 26, 29, 30, 55, 55, 114, 41, 41, 60, 60, 60, 60, 57, 57, 57, 57, 57, 29, 58, 58, 58, 58, 57, 13, 62, 62, 49, 47, 47, 47, 47, 47, 58, 18, 30, 30, 30, 30, 60, 22, 22, 24, 51, 27, 21, 21, 21, 20, 51, 51, 15, 17, 32, 32, 23, 51, 72, 72, 27, 70, 48, 64, 48, 48, 18, 35, 26, 43, 26, 41, 34, 21, 21, 21, 31, 26, 30, 30, 48, 48, 47, 47, 47, 39, 39, 39, 39, 20, 57, 37, 43, 43, 43, 43, 27, 27, 27, 57, 27, 57, 62, 62, 22, 34, 34, 84, 58, 26, 52, 52, 85, 101, 101, 101, 101, 101, 84, 84, 84, 49, 49, 49, 50, 50, 50, 50, 50, 50, 50, 50, 20, 42, 19, 33, 64, 64, 64, 8, 25, 48, 48, 92, 20, 42, 42, 42, 42, 60, 60, 60, 60, 28, 27, 38, 38, 72, 17, 25, 51, 16, 25, 51, 48, 33, 68, 29, 29, 29, 29, 29, 42, 73, 73, 73, 73, 73, 73, 45, 45, 45, 45, 45, 21, 45, 45, 45, 30, 30, 30, 73, 44, 44, 24, 24, 24, 70, 70, 70, 60, 60, 60, 69, 69, 30, 24, 24, 13, 53, 53, 53, 53, 29, 29, 29, 27, 26, 26, 26, 26, 28, 28, 28, 28, 28, 28, 22, 32, 30, 66, 66, 66, 66, 26, 33, 30, 30, 20, 28, 69, 69, 86, 69, 11, 23, 40, 40, 42, 42, 14, 23, 18, 47, 47, 47, 47, 25, 25, 25, 25, 24, 24, 18, 19, 26, 93, 15, 34, 26, 32, 12, 17, 17, 35, 35, 20, 42, 42, 42, 16, 16, 19, 46, 27, 22, 14, 17, 38, 29, 38, 29, 45, 45, 51, 20, 37, 37, 37, 37, 14, 52, 68, 29, 42, 27, 38, 27, 30, 37, 37, 37, 37, 29, 68, 57, 13, 13, 16, 23, 13, 29, 21, 13, 13, 37, 11, 58, 58, 58, 26, 26, 46, 46, 46, 21, 133, 51, 51, 54, 54, 54, 25, 40, 40, 13, 32, 23, 23, 11, 18, 52, 18, 55, 55, 45, 45, 16, 25, 25, 11, 40, 56, 80, 80, 35, 35, 35, 56, 56, 68, 68, 104, 16, 34, 23, 33, 40, 21, 38, 19, 41, 41, 33, 22, 73, 73, 73, 86, 33, 28, 40, 40, 41, 54, 28, 37, 27, 47, 36, 36, 36, 39, 70, 70, 70, 70, 40, 19, 99, 99, 99, 99, 99, 99, 28, 28, 17, 26, 54, 37, 36, 23, 36, 84, 29, 26, 48, 45, 65, 31, 21, 26, 26, 26, 35, 64, 64, 34, 35, 53, 31, 31, 38, 60, 27, 38, 74, 29, 29, 37, 82, 14, 14, 37, 37, 37, 37, 20, 42, 42, 42, 42, 42, 44, 44, 67, 26, 47, 47, 47, 85, 32, 81, 49, 49, 49, 26, 52, 52, 52, 52, 21, 21, 21, 34, 34, 34, 88, 23, 71, 25, 25, 28, 28, 22, 54, 54, 25, 30, 10, 24, 30, 30, 16, 32, 32, 32, 46, 46, 46, 26, 26, 48, 48, 38, 55, 55, 38, 38, 52, 52, 21, 52, 29, 21, 75, 21, 49, 52, 75, 88, 88, 52, 33, 33, 28, 31, 39, 19, 19, 29, 29, 32, 19, 14, 90, 34, 34, 50, 50, 32, 32, 14, 31, 31, 31, 31, 31, 29, 29, 40, 25, 25, 37, 21, 35, 24, 24, 26, 26, 27, 9, 27, 58, 45, 45, 10, 10, 44, 76, 83, 83, 44, 76, 83, 83, 28, 48, 48, 20, 37, 37, 37, 67, 95, 28, 28, 28, 28, 58, 17, 75, 27, 39, 17, 75, 27, 39, 29, 29, 37, 37, 18, 29, 19, 19, 101, 82, 101, 82, 82, 82, 82, 82, 82, 17, 38, 82, 101, 82, 82, 35, 35, 24, 59, 21, 36, 36, 33, 69, 33, 69, 33, 59, 64, 50, 52, 52, 21, 26, 18, 71, 26, 45, 32, 33, 33, 53, 53, 53, 53, 53, 39, 39, 39, 39, 42, 42, 42, 23, 17, 29, 42, 72, 72, 47, 30, 32, 58, 58, 38, 50, 34, 20, 39, 55, 63, 42, 43, 15, 19, 16, 45, 45, 26, 43, 26, 44, 37, 36, 36, 32, 47, 47, 47, 83, 18, 25, 52, 27, 30, 76, 31, 26, 17, 38, 38, 20, 19, 19, 41, 41, 41, 55, 66, 66, 36, 65, 65, 65, 65, 14, 27, 27, 27, 27, 20, 20, 58, 26, 26, 25, 30, 47, 30, 30, 75, 26, 32, 32, 32, 32, 38, 38, 38, 38, 38, 15, 28, 28, 28, 35, 26, 26, 49, 49, 49, 49, 49, 49, 33, 33, 33, 50, 62, 62, 28, 70, 29, 29, 35, 35, 26, 15, 41, 41, 87, 87, 215, 109, 109, 109, 109, 109, 109, 35, 84, 84, 64, 13, 39, 39, 39, 64, 39, 33, 22, 97, 97, 97, 97, 97, 38, 61, 36, 25, 38, 128, 128, 128, 128, 68, 118, 20, 50, 65, 12, 23, 18, 16, 40, 18, 18, 12, 21, 21, 10, 22, 22, 26, 26, 13, 26, 26, 16, 80, 80, 14, 24, 22, 34, 34, 34, 11, 24, 23, 23, 23, 21, 21, 44, 44, 44, 21, 28, 19, 28, 42, 54, 54, 21, 18, 33, 20, 56, 40, 40, 36, 59, 27, 30, 34, 34, 34, 24, 24, 24, 24, 24, 28, 38, 38, 22, 39, 39, 79, 79, 79, 94, 94, 52, 81, 29, 16, 23, 23, 21, 19, 19, 29, 29, 22, 12, 22, 15, 37, 37, 27, 27, 51, 51, 51, 53, 53, 53, 74, 53, 53, 53, 74, 53, 53, 53, 74, 19, 30, 11, 30, 78, 78, 78, 78, 44, 44, 44, 25, 49, 49, 35, 35, 28, 28, 28, 28, 51, 30, 30, 25, 25, 24, 24, 31, 31, 27, 27, 38, 38, 38, 28, 28, 38, 38, 27, 27, 47, 25, 25, 12, 12, 37, 37, 29, 29, 29, 29, 52, 52, 52, 52, 36, 36, 36, 36, 30, 30, 30, 30, 45, 45, 45, 35, 35, 35, 52, 31, 31, 27, 27, 39, 24, 24, 30, 30, 28, 28, 45, 28, 28, 40, 40, 40, 35, 35, 35, 39, 39, 31, 31, 35, 16, 44, 44, 44, 33, 33, 33, 39, 39, 39, 39, 52, 52, 90, 90, 30, 30, 30, 46, 28, 28, 49, 49, 64, 24, 35, 35, 70, 70, 23, 25, 25, 13, 35, 20, 39, 39, 39, 39, 23, 36, 30, 22, 49, 24, 24, 17, 41, 28, 28, 23, 37, 37, 43, 20, 37, 34, 28, 28, 39, 39, 39, 39, 27, 27, 60, 66, 66, 76, 76, 58, 58, 54, 54, 54, 33, 25, 25, 42, 42, 70, 42, 30, 17, 17, 46, 46, 21, 50, 50, 52, 52, 36, 19, 19, 19, 24, 24, 24, 18, 30, 30, 30, 57, 34, 31, 29, 31, 44, 44, 28, 28, 31, 28, 21, 30, 40, 37, 59, 59, 37, 18, 18, 19, 21, 21, 56, 56, 47, 47, 47, 78, 66, 66, 27, 60, 51, 51, 51, 51, 44, 44, 44, 44, 93, 93, 93, 93, 93, 93, 93, 39, 39, 39, 50, 73, 18, 18, 18, 46, 113, 113, 113, 113, 113, 113, 113, 57, 42, 40, 15, 33, 33, 33, 23, 23, 55, 23, 23, 23, 24, 16, 16, 12, 20, 20, 11, 18, 18, 63, 40, 23, 23, 72, 63, 23, 29, 29, 16, 16, 39, 39, 39, 39, 64, 64, 42, 81, 42, 57, 69, 57, 12, 18, 24, 24, 37, 17, 17, 52, 34, 34, 34, 34, 34, 69, 69, 69, 69, 69, 69, 69, 69, 69, 35, 35, 30, 30, 30, 30, 69, 69, 21, 21, 21, 29, 65, 65, 65, 34, 34, 34, 34, 34, 19, 34, 21, 21, 36, 30, 35, 35, 40, 40, 40, 40, 27, 25, 25, 33, 23, 33, 33, 34, 34, 26, 33, 33, 33, 8, 24, 60, 18, 18, 38, 38, 21, 30, 30, 17, 16, 43, 35, 30, 30, 35, 23, 24, 24, 17, 17, 45, 45, 82, 82, 82, 82, 55, 93, 100, 55, 93, 100, 55, 93, 100, 16, 28, 48, 20, 71, 30, 39, 39, 29, 29, 39, 39, 26, 30, 30, 16, 41, 41, 25, 25, 10, 88, 58, 65, 65, 65, 65, 65, 65, 65, 38, 18, 25, 25, 13, 27, 27, 65, 65, 65, 14, 29, 29, 29, 39, 39, 39, 20, 25, 33, 53, 20, 17, 19, 51, 51, 51, 51, 51, 38, 21, 23, 15, 32, 34, 39, 39, 39, 45, 25, 22, 22, 39, 39, 39, 27, 27, 24, 19, 36, 28, 75, 47, 47, 13, 61, 30, 20, 62, 58, 16, 24, 15, 8, 39, 39, 52, 68, 90, 43, 49, 49, 81, 68, 120, 120, 120, 78, 78, 78, 78, 92, 78, 108, 108, 108, 108, 108, 147, 108, 59, 59, 59, 59, 86, 86, 86, 101, 40, 27, 27, 27, 26, 18, 39, 39, 81, 81, 81, 39, 42, 35, 15, 47, 32, 26, 26, 85, 85, 85, 121, 121, 121, 46, 69, 43, 43, 43, 43, 74, 103, 74, 103, 74, 74, 74, 34, 34, 34, 34, 45, 11, 29, 29, 27, 49, 93, 26, 26, 13, 36, 29, 26, 40, 40, 33, 33, 33, 65, 23, 30, 31, 30, 30, 25, 18, 45, 27, 25, 30, 17, 98, 98, 98, 17, 27, 132, 105, 105, 105, 114, 133, 19, 44, 28, 28, 28, 35, 18, 24, 63, 63, 63, 111, 94, 132, 20, 30, 70, 49, 44, 28, 54, 54, 34, 25, 25, 16, 35, 63, 42, 47, 47, 39, 14, 40, 40, 31, 28, 20, 36, 41, 14, 18, 16, 61, 21, 32, 18, 75, 68, 20, 22, 50, 50, 94, 94, 94, 114, 38, 29, 38, 62, 62, 78, 31, 17, 39, 29, 80, 57, 21, 21, 37, 37, 64, 26, 17, 85, 63, 23, 23, 21, 36, 30, 51, 32, 54, 31, 25, 32, 32, 23, 20, 20, 56, 50, 50, 22, 67, 48, 37, 37, 67, 67, 29, 59, 59, 59, 59, 59, 59, 59, 41, 28, 28, 51, 51, 33, 48, 18, 19, 65, 36, 68, 83, 118, 27, 122, 106, 27, 32, 32, 27, 27, 23, 63, 56, 56, 56, 79, 66, 17, 34, 34, 26, 46, 14, 40, 40, 40, 20, 20, 25, 87, 87, 87, 87, 87, 87, 14, 51, 51, 51, 51, 51, 47, 47, 47, 47, 20, 26, 18, 20, 20, 15, 84, 84, 84, 84, 101, 84, 22, 22, 14, 19, 41, 19, 46, 46, 46, 46, 46, 90, 90, 90, 90, 90, 90, 103, 103, 41, 41, 41, 41, 41, 37, 37, 37, 37, 37, 46, 46, 76, 76, 76, 32, 32, 32, 32, 13, 16, 16, 39, 39, 39, 39, 12, 12, 20, 37, 50, 74, 74, 37, 37, 41, 41, 53, 41, 21, 22, 95, 95, 95, 95, 95, 22, 16, 31, 31, 16, 31, 122, 122, 122, 23, 23, 23, 40, 40, 40, 13, 76, 76, 76, 36, 86, 16, 36, 86, 37, 37, 37, 37, 35, 35, 23, 23, 23, 27, 27, 47, 47, 27, 27, 156, 156, 25, 37, 37, 37, 37, 37, 33, 33, 15, 15, 21, 21, 21, 21, 7, 28, 25, 25, 141, 141, 31, 31, 21, 21, 31, 31, 99, 123, 36, 36, 36, 36, 36, 36, 14, 32, 32, 44, 14, 47, 35, 35, 20, 54, 34, 34, 34, 65, 65, 65, 65, 45, 45, 45, 45, 25, 52, 52, 25, 52, 52, 52, 52, 52, 52, 52, 52, 21, 21, 41, 41, 41, 46, 66, 66, 66, 46, 46, 46, 46, 46, 46, 38, 38, 13, 42, 42, 42, 42, 24, 24, 24, 28, 46, 140, 24, 140, 24, 140, 24, 54, 54, 31, 28, 28, 28, 47, 47, 47, 29, 27, 27, 52, 32, 31, 26, 23, 26, 47, 43, 43, 19, 19, 28, 28, 15, 78, 78, 78, 19, 30, 30, 30, 41, 41, 49, 49, 49, 26, 26, 20, 42, 42, 42, 42, 42, 85, 85, 82, 82, 82, 82, 82, 20, 25, 41, 29, 130, 95, 95, 95, 12, 37, 37, 37, 78, 78, 78, 66, 53, 30, 42, 30, 42, 23, 23, 23, 88, 88, 106, 115, 142, 142, 142, 12, 39, 39, 135, 39, 135, 39, 39, 135, 39, 135, 41, 41, 27, 27, 35, 49, 35, 49, 140, 20, 120, 65, 45, 45, 45, 61, 20, 61, 20, 27, 42, 27, 27, 42, 27, 40, 40, 39, 39, 20, 40, 49, 154, 176, 195, 49, 154, 176, 195, 42, 42, 42, 42, 33, 20, 30, 85, 165, 234, 27, 38, 38, 38, 38, 30, 30, 30, 85, 53, 32, 32, 85, 85, 46, 31, 40, 59, 28, 28, 38, 38, 38, 40, 90, 90, 96, 118, 50, 50, 24, 44, 66, 19, 23, 23, 26, 29, 29, 29, 32, 51, 32, 51, 20, 18, 18, 15, 15, 15, 50, 50, 50, 50, 29, 50, 24, 181, 181, 217, 259, 111, 61, 46, 46, 28, 28, 28, 25, 54, 44, 44, 42, 86, 86, 47, 28, 34, 34, 28, 28, 42, 86, 28, 34, 32, 33, 10, 20, 58, 54, 54, 54, 24, 29, 22, 22, 22, 22, 50, 23, 48, 59, 59, 17, 36, 57, 57, 141, 62, 126, 126, 62, 126, 126, 65, 40, 82, 128, 36, 36, 36, 36, 80, 80, 80, 80, 80, 80, 33, 39, 15, 69, 69, 69, 57, 29, 29, 29, 29, 27, 27, 39, 51, 21, 21, 50, 50, 25, 101, 101, 101, 101, 28, 47, 47, 13, 34, 23, 27, 27, 34, 31, 24, 24, 27, 32, 32, 17, 55, 46, 14, 24, 33, 33, 24, 24, 24, 51, 15, 26, 26, 15, 50, 15, 50, 33, 33, 34, 50, 58, 34, 50, 58, 104, 104, 104, 76, 76, 76, 51, 33, 33, 57, 29, 57, 29, 53, 48, 48, 48, 48, 25, 25, 25, 16, 39, 39, 46, 58, 22, 37, 37, 18, 18, 23, 23, 52, 23, 39, 39, 39, 39, 15, 15, 30, 12, 12, 16, 16, 13, 37, 37, 64, 45, 45, 66, 10, 24, 24, 24, 24, 30, 39, 30, 30, 39, 59, 59, 63, 63, 63, 63, 63, 63, 18, 40, 26, 48, 26, 48, 40, 40, 40, 44, 44, 59, 87, 44, 70, 70, 70, 70, 70, 186, 186, 186, 186, 186, 186, 186, 186, 34, 34, 21, 21, 34, 66, 66, 103, 37, 83, 83, 83, 46, 10, 81, 30, 30, 136, 16, 16, 31, 31, 22, 43, 58, 43, 19, 22, 25, 38, 38, 22, 32, 32, 36, 36, 41, 63, 57, 24, 24, 91, 118, 25, 27, 112, 70, 35, 35, 21, 21, 65, 49, 49, 67, 27, 143, 35, 108, 108, 19, 19, 30, 66, 66, 98, 135, 31, 24, 23, 23, 51, 28, 33, 19, 73, 94, 67, 14, 43, 43, 31, 17, 25, 35, 35, 69, 69, 18, 100, 42, 26, 58, 58, 43, 21, 16, 22, 59, 20, 51, 22, 22, 41, 41, 58, 14, 14, 14, 32, 43, 55, 71, 32, 28, 39, 54, 14, 5, 12, 19, 41, 64, 67, 67, 18, 12, 11, 58, 73, 94, 58, 67, 13, 20, 20, 20, 37, 12, 30, 30, 30, 29, 29, 46, 46, 32, 44, 44, 44, 44, 31, 16, 22, 82, 82, 82, 82, 82, 82, 24, 21, 21, 22, 22, 56, 56, 26, 42, 42, 42, 42, 36, 22, 9, 10, 17, 35, 15, 20, 25, 35, 71, 71, 33, 39, 39, 94, 94, 21, 21, 32, 36, 36, 22, 36, 36, 41, 41, 35, 35, 26, 20, 34, 18, 13, 16, 20, 12, 30, 30, 11, 25, 5, 6, 23, 23, 23, 21, 17, 32, 32, 32, 32, 32, 32, 20, 20, 19, 19, 39, 12, 23, 23, 34, 47, 27, 43, 24, 24, 26, 22, 25, 71, 71, 89, 18, 76, 76, 76, 76, 20, 90, 90, 90, 90, 90, 90, 22, 14, 31, 31, 31, 69, 39, 69, 39, 69, 39, 90, 39, 23, 26, 60, 60, 11, 12, 58, 58, 58, 58, 11, 23, 23, 31, 31, 35, 35, 61, 61, 61, 61, 23, 51, 27, 19, 23, 42, 158, 104, 23, 42, 54, 54, 22, 22, 16, 39, 39, 39, 19, 23, 21, 23, 19, 29, 28, 28, 28, 52, 84, 84, 56, 22, 22, 34, 26, 26, 25, 37, 36, 24, 39, 91, 36, 36, 24, 34, 24, 44, 44, 44, 44, 44, 44, 50, 75, 75, 50, 45, 43, 29, 29, 21, 29, 19, 19, 19, 66, 91, 77, 77, 77, 27, 88, 88, 23, 39, 27, 21, 36, 36, 36, 36, 60, 60, 35, 35, 20, 39, 39, 50, 39, 25, 29, 29, 29, 29, 29, 27, 27, 38, 26, 26, 29, 35, 35, 35, 35, 32, 32, 46, 46, 48, 29, 29, 12, 25, 24, 64, 14, 21, 20, 38, 22, 22, 23, 23, 23, 28, 28, 23, 63, 24, 56, 36, 36, 24, 56, 36, 36, 74, 112, 24, 24, 67, 40, 22, 61, 61, 63, 63, 63, 63, 63, 63, 37, 33, 20, 50, 30, 25, 16, 78, 78, 78, 78, 38, 30, 32, 32, 32, 21, 24, 42, 42, 42, 42, 24, 18, 18, 34, 34, 12, 30, 36, 36, 15, 15, 51, 51, 51, 51, 51, 27, 27, 29, 29, 29, 27, 20, 20, 36, 29, 14, 13, 16, 16, 16, 35, 35, 35, 35, 42, 51, 19, 19, 19, 27, 32, 37, 49, 15, 13, 60, 40, 40, 40, 19, 59, 62, 11, 62, 62, 52, 52, 52, 57, 64, 53, 16, 59, 53, 78, 11, 18, 47, 17, 92, 69, 69, 69, 69, 69, 69, 69, 43, 60, 60, 60, 60, 60, 73, 11, 15, 26, 15, 17, 17, 17, 17, 34, 34, 34, 29, 30, 30, 12, 32, 32, 32, 44, 50, 64, 32, 23, 27, 13, 22, 22, 31, 22, 15, 14, 15, 11, 21, 12, 31, 31, 31, 49, 49, 82, 32, 49, 49, 82, 49, 49, 82, 18, 17, 15, 19, 32, 32, 43, 63, 4, 13, 13, 11, 20, 20, 32, 18, 21, 31, 31, 10, 22, 22, 22, 11, 21, 26, 26, 13, 27, 27, 27, 39, 27, 39, 19, 47, 47, 47, 28, 28, 15, 14, 12, 12, 34, 34, 34, 13, 13, 20, 31, 19, 39, 74, 27, 16, 53, 58, 58, 13, 45, 32, 43, 14, 75, 51, 24, 25, 52, 52, 15, 15, 25, 39, 18, 18, 18, 39, 39, 39, 39, 39, 51, 20, 25, 20, 37, 45, 93, 125, 76, 108, 16, 16, 68, 68, 68, 46, 46, 61, 61, 61, 61, 20, 10, 77, 67, 67, 42, 79, 79, 42, 79, 79, 42, 39, 45, 45, 35, 35, 35, 26, 19, 19, 18, 26, 26, 12, 12, 37, 37, 37, 40, 40, 40, 40, 40, 40, 40, 40, 96, 96, 39, 39, 21, 21, 30, 30, 30, 41, 41, 41, 41, 41, 41, 75, 59, 10, 26, 26, 38, 24, 44, 44, 24, 20, 20, 15, 15, 53, 53, 38, 38, 46, 92, 92, 109, 339, 384, 46, 46, 46, 46, 46, 46, 46, 46, 36, 45, 29, 58, 58, 105, 29, 29, 30, 30, 30, 30, 30, 69, 106, 119, 106, 119, 106, 119, 106, 119, 106, 119, 106, 119, 23, 36, 13, 29, 29, 29, 29, 29, 29, 18, 18, 40, 42, 21, 17, 15, 27, 60, 70, 70, 70, 31, 67, 67, 47, 26, 26, 43, 27, 27, 115, 69, 69, 69, 27, 27, 27, 27, 63, 63, 63, 63, 63, 63, 63, 24, 20, 45, 85, 85, 38, 45, 45, 45, 45, 45, 49, 49, 49, 93, 53, 53, 70, 70, 42, 42, 42, 42, 45, 70, 70, 25, 22, 22, 30, 67, 67, 28, 32, 31, 66, 161, 66, 86, 86, 86, 86, 86, 28, 28, 39, 123, 67, 84, 67, 67, 84, 84, 84, 84, 67, 84, 84, 84, 67, 22, 86, 52, 29, 48, 45, 22, 18, 13, 17, 17, 42, 31, 31, 81, 81, 81, 81, 81, 81, 81, 81, 16, 54, 54, 67, 49, 49, 49, 19, 46, 35, 86, 86, 86, 86, 38, 38, 62, 86, 86, 96, 96, 96, 57, 57, 57, 82, 88, 88, 46, 46, 46, 23, 60, 60, 60, 60, 11, 37, 23, 100, 100, 100, 100, 100, 87, 118, 87, 87, 39, 71, 71, 71, 22, 37, 66, 99, 99, 46, 71, 90, 30, 30, 39, 37, 36, 23, 59, 148, 58, 90, 90, 98, 52, 28, 81, 34, 34, 34, 34, 81, 46, 35, 35, 35, 103, 103, 45, 45, 39, 39, 61, 61, 79, 79, 79, 25, 25, 25, 22, 28, 9, 78, 78, 78, 78, 46, 20, 26, 63, 63, 42, 90, 90, 90, 90, 90, 90, 90, 125, 125, 125, 125, 40, 21, 52, 62, 36, 36, 52, 62, 15, 51, 51, 51, 51, 27, 27, 44, 44, 41, 35, 46, 46, 46, 46, 31, 31, 31, 44, 31, 31, 53, 53, 53, 23, 39, 13, 11, 35, 23, 26, 41, 41, 41, 58, 30, 22, 17, 31, 31, 24, 24, 47, 47, 29, 38, 38, 66, 66, 66, 48, 48, 43, 43, 43, 43, 43, 49, 56, 103, 56, 103, 85, 85, 91, 47, 47, 18, 32, 32, 20, 42, 117, 85, 85, 58, 27, 39, 27, 39, 75, 22, 58, 58, 77, 77, 44, 44, 19, 36, 39, 52, 39, 31, 39, 52, 67, 18, 38, 56, 20, 25, 38, 13, 18, 82, 116, 13, 64, 98, 51, 66, 51, 19, 19, 18, 53, 53, 53, 53, 53, 65, 65, 65, 65, 45, 45, 45, 45, 19, 45, 45, 45, 45, 13, 13, 36, 36, 36, 20, 38, 38, 38, 38, 8, 48, 34, 34, 34, 13, 27, 20, 34, 20, 14, 34, 17, 27, 27, 34, 28, 33, 33, 33, 18, 23, 36, 36, 30, 30, 30, 42, 42, 42, 42, 42, 42, 17, 17, 42, 24, 78, 78, 29, 29, 29, 20, 41, 41, 41, 48, 48, 48, 48, 48, 48, 33, 33, 33, 33, 75, 75, 75, 86, 29, 29, 27, 27, 31, 43, 43, 70, 22, 30, 30, 30, 60, 21, 13, 20, 14, 20, 35, 35, 101, 42, 42, 42, 26, 26, 51, 33, 40, 33, 33, 38, 38, 38, 38, 28, 77, 77, 23, 38, 38, 119, 119, 119, 38, 33, 33, 38, 103, 103, 103, 37, 40, 40, 55, 58, 58, 24, 31, 23, 23, 23, 59, 27, 82, 82, 21, 60, 60, 60, 60, 40, 40, 40, 40, 50, 50, 23, 69, 69, 69, 24, 23, 29, 44, 44, 20, 38, 42, 42, 42, 46, 46, 46, 46, 46, 46, 46, 46, 53, 53, 41, 41, 41, 60, 60, 67, 72, 14, 18, 41, 31, 85, 27, 27, 27, 27, 101, 101, 101, 44, 57, 57, 57, 21, 30, 46, 15, 28, 28, 28, 28, 17, 28, 28, 28, 28, 28, 28, 64, 64, 64, 64, 18, 29, 37, 37, 37, 13, 61, 61, 61, 61, 61, 30, 30, 41, 25, 21, 21, 68, 68, 68, 68, 68, 68, 21, 52, 64, 77, 17, 38, 38, 38, 38, 38, 18, 56, 42, 55, 37, 37, 37, 84, 56, 56, 84, 84, 84, 24, 24, 19, 21, 13, 24, 34, 36, 16, 14, 23, 23, 24, 24, 24, 24, 36, 36, 36, 36, 21, 54, 66, 38, 19, 19, 42, 42, 21, 53, 53, 24, 74, 25, 23, 31, 31, 31, 35, 16, 22, 43, 43, 43, 24, 27, 33, 18, 18, 25, 25, 30, 63, 31, 31, 31, 58, 58, 67, 13, 36, 36, 36, 36, 33, 70, 22, 34, 16, 12, 31, 38, 46, 65, 46, 65, 13, 22, 22, 53, 53, 53, 53, 53, 21, 21, 15, 15, 44, 44, 63, 70, 97, 44, 33, 33, 33, 12, 19, 16, 22, 27, 27, 21, 45, 45, 45, 54, 54, 54, 20, 20, 20, 9, 33, 41, 33, 11, 12, 47, 27, 15, 25, 30, 30, 30, 30, 30, 46, 46, 46, 46, 24, 24, 24, 24, 11, 11, 47, 21, 31, 31, 31, 21, 66, 45, 45, 45, 45, 58, 35, 35, 35, 35, 35, 26, 26, 61, 32, 32, 35, 32, 32, 19, 19, 24, 10, 16, 58, 58, 58, 70, 58, 58, 69, 69, 23, 38, 24, 36, 15, 21, 51, 51, 51, 51, 51, 29, 43, 26, 33, 23, 51, 37, 29, 36, 42, 27, 20, 11, 29, 29, 23, 23, 37, 24, 29, 40, 40, 23, 43, 43, 51, 36, 27, 35, 43, 26, 26, 36, 22, 22, 30, 30, 30, 29, 29, 29, 26, 26, 36, 36, 36, 41, 10, 36, 36, 36, 32, 21, 23, 23, 27, 12, 39, 26, 27, 13, 27, 42, 29, 29, 31, 22, 18, 20, 24, 29, 18, 25, 125, 125, 76, 76, 76, 26, 53, 86, 34, 19, 52, 52, 53, 53, 53, 53, 53]
---句子长度列表长度: 8124
---最长句子字符个数： 384 最短句子字符个数： 4
---数据数量： 8124
---train_split_spot: 6499
---dev_split_spot: 7311
---train_list的数量： 6499
---dev_list的数量： 812
---test_list的数量： 813
'''
import random
def data_split(diakg_have_re_ok_path,train_rate,dev_rate,test_rate):
    '''读取CMeIE_ok_path'''
    with open(diakg_have_re_ok_path, 'r', encoding='utf-8') as f:
        all_list = f.readlines()
    f.close()

    '''查看一下句子的长度'''
    len_list = []
    for sentence in all_list:
        sentence = eval(sentence)
        len_list.append(len(sentence['token']))
    print("---句子长度列表:",len_list)
    print("---句子长度列表长度:",len(len_list))
    print("---最长句子字符个数：",max(len_list),"最短句子字符个数：",min(len_list))
    # print(all_list)
    print("---数据数量：",len(all_list))
    random.shuffle(all_list)
    # print(all_list)

    '''训练集的切分点'''
    train_split_spot = int((train_rate / (train_rate + dev_rate + test_rate)) * len(all_list))
    dev_split_spot = int(train_split_spot+((len(all_list)-train_split_spot)/2))
    print("---train_split_spot:",train_split_spot)
    print("---dev_split_spot:",dev_split_spot)

    '''切分'''
    train_list = all_list[:train_split_spot]
    dev_list = all_list[train_split_spot:dev_split_spot]
    test_list = all_list[dev_split_spot:]
    print("---train_list的数量：",len(train_list))
    print("---dev_list的数量：",len(dev_list))
    print("---test_list的数量：",len(test_list))
    return train_list,dev_list,test_list



'''以8：1：1对数据进行切分'''
train_rate = 8
dev_rate = 1
test_rate = 1
diakg_have_re_ok_path = '../datasets/diakg/diakg_have_re_ok.txt'
train_list,dev_list,test_list = data_split(diakg_have_re_ok_path, train_rate, dev_rate, test_rate)

'''保存数据'''
diakg_without_no_re_train_path = '../datasets/diakg_without_no_re/train.txt'
with open(diakg_without_no_re_train_path,'w+',encoding='utf-8') as f:
    for i in train_list:
        f.write(str(i))
f.close()

diakg_without_no_re_dev_path = '../datasets/diakg_without_no_re/dev.txt'
with open(diakg_without_no_re_dev_path,'w+',encoding='utf-8') as f:
    for i in dev_list:
        f.write(str(i))
f.close()

diakg_without_no_re_test_path = '../datasets/diakg_without_no_re/test.txt'
with open(diakg_without_no_re_test_path,'w+',encoding='utf-8') as f:
    for i in test_list:
        f.write(str(i))
f.close()
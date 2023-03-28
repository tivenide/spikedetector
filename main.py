
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Strg+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

#for one recording
path='data/'

from tools import preprocessing_for_one_recording
preprocessing_for_one_recording(path)

from visualization import visualize_assignments_of_one_recording
visualize_assignments_of_one_recording(path, 100)

print('finish')
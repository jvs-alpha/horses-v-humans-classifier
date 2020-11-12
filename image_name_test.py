from hvmtest import *


test, t_label = loadtest()
class_names = ["horse","human"]
for index in range(len(test)):
    plt.imshow(test[index], cmap="gray")
    plt.title(class_names[t_label[index]])
    plt.show()

import matplotlib.pyplot as plt
a_list = list(range(0, 20))
b_list = list(range(0, 100, 5))

plt.rcParams['axes.linewidth'] = 1.0  # axis line width
plt.rcParams["font.size"] = 24  # 全体のフォントサイズが変更されます。
plt.rcParams['axes.grid'] = True  # make grid
plt.plot(a_list, linewidth=2.0, marker='o')
plt.plot(b_list, linewidth=2.0, marker='o')
plt.title('model loss')
plt.xlabel('epoch', fontsize=18)
plt.ylabel('loss', fontsize=18)
plt.tick_params(labelsize=20)
plt.legend(['loss', 'val_loss'], loc='upper right', fontsize=18)
plt.tight_layout()
plt.savefig('test.png')

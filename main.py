import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy


def plotKappaErrorGraph(fileName, exp):
    colors = cm.rainbow(numpy.linspace(0, 1, 10))
    markers = ['.', 'v', '<', '2', 'p', '*', 'X', 's']
    _ = plt.figure(dpi=250)
    plt.grid()

    plots = []
    # for i in range(1, 8):
    for i in range(1, len(exp) + 1):
        # path = f'run/exp{i}/KappaErrorGraphPoints.txt'
        # path = f'experiments/2-3/exp{i}/KappaErrorGraphPoints.txt'
        path = f'rapor/tmp/exp{i}/KappaErrorGraphPoints.txt'
        with open(path, 'r') as f:
            lines = f.readlines()
        f.close()

        x_points = lines[0].replace('\n', '').split(', ')
        y_points = lines[1].replace('\n', '').split(', ')

        x = [float(p) for p in x_points]
        y = [float(p) for p in y_points]

        plot = plt.scatter(x, y, s=10, color=colors[i - 1], marker=markers[i - 1], alpha=0.5)
        plots.append(plot)

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # plt.title(' Kappa-Error Graph')
    plt.xlabel(' κ ')
    plt.ylabel(' Error ')

    plt.legend(plots, exp)

    plt.savefig(f'{fileName}.png')
    plt.close()
    return


if __name__ == '__main__':
    # exp = ['SGD-CyclicLR- α = 0',
    #        'SGD-MSLR- α = 0',
    #        'Adam-MSLR- α = 0',
    #        'Adam-CyclicLR- α = 0',
    #        'Adam-CyclicLR- α = 0.1',
    #        'Adam-CyclicLR- α = 0.2',
    #        'Adam-CyclicLR- α = 0.3']
    # plotKappaErrorGraph('Kappa-ErrorGraph-Deney1', exp)

    # exp = ['w/o Self-Training',
    #        'α = 0',
    #        'α = 0.1',
    #        'α = 0.2',
    #        'α = 0.3']
    # plotKappaErrorGraph('Kappa-ErrorGraph-Deney2', exp)

    # exp = ['w/o Self-Training',
    #        'w/ Self-Training',
    #        'w/ Self-Training(%60)']
    # plotKappaErrorGraph('Kappa-ErrorGraph-Deney3-4-1', exp)

    # exp = ['w/o Self-Training',
    #        'w/ Self-Training',
    #        'w/ Self-Training(%60)']
    # plotKappaErrorGraph('Kappa-ErrorGraph-Deney3-4-2', exp)

    # exp = ['w/o Self-Training',
    #        'w/ Self-Training',
    #        'w/ Self-Training(%60)',
    #        'w/ Self-Training(%80)',
    #        'w/ Self-Training(%80)']
    # plotKappaErrorGraph('Kappa-ErrorGraph-Deney4', exp)

    # exp = ['w/o Self-Training',
    #        'w/ Self-Training',
    #        'w/ Self-Training(%60)',
    #        'w/ Self-Training(first %80)',
    #        'w/ Self-Training(last %80)']
    # plotKappaErrorGraph('Kappa-ErrorGraph-Deney5', exp)

    # exp = ['w/o Self-Training',
    #        'w/ Self-Training',
    #        'w/ Self-Training(%60)',
    #        'w/ Self-Training(first %80)',
    #        'w/ Self-Training(last %80)']
    # plotKappaErrorGraph('Kappa-ErrorGraph-Deney4', exp)

    # exp = ['w/o Self-Training, w/o K-Fold',
    #        'w/o Self-Training, w/ K-Fold',
    #        'w/ Self-Training, w/ K-Fold']
    # plotKappaErrorGraph('Kappa-ErrorGraph-Deney3', exp)
    #
    # exp = ['w/o Self-Training',
    #        'w/ Self-Training',
    #        'w/ Self-Training(%0-%20)',
    #        'w/ Self-Training(%0-%30)',
    #        'w/ Self-Training(%0-%40)',
    #        'w/ Self-Training(%20-%80)',
    #        'w/ Self-Training(%20-%100)',
    #        'w/ Self-Training(%0-%80)']
    # plotKappaErrorGraph('Kappa-ErrorGraph-Deney4', exp)

    # exp = ['w/o Self-Training',
    #        'w/ Self-Training',
    #        'w/ Self-Training(%0-%20)',
    #        'w/ Self-Training(%0-%30)',
    #        'w/ Self-Training(%0-%40)',
    #        'w/ Self-Training(%20-%80)',
    #        'w/ Self-Training(%0-%80)']
    #
    # plotKappaErrorGraph('Kappa-ErrorGraph-Deney4-v2', exp)

    exp = ['w/o Self-Training',
           'w/ Self-Training',
           'w/ Self-Training(%0-%20)',
           'w/ Self-Training(%0-%30)',
           'w/ Self-Training(%0-%40)',
           'w/ Self-Training(%20-%80)',
           'w/ Self-Training(%0-%80)']

    plotKappaErrorGraph('Kappa-ErrorGraph-Deney5-v2', exp)

    # exp = ['w/o Self-Training',
    #        'w/ Self-Training',
    #        'w/ Self-Training(%60)']
    # plotKappaErrorGraph('Kappa-ErrorGraph-Deney5', exp)

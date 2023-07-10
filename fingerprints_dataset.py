import numpy
import matplotlib
import matplotlib.pyplot as plt

def mcol(v):
    return v.reshape((v.size, 1))

def load(fname):
    DList = []
    labelsList = []
    hLabels = {
        'Authentic-fingerprint': 1,
        'Spoofed-fingerprint': 0,
        
        }

    with open(fname) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:10]
                attrs = mcol(numpy.array([float(i) for i in attrs]))
                label = int(line.split(',')[-1].strip())
                
                DList.append(attrs)
                labelsList.append(label)
            except:
                pass

    return numpy.hstack(DList), numpy.array(labelsList, dtype=numpy.int32)




def plot_hist(D, L):

    D0 = D[:, L==0]
    D1 = D[:, L==1]
    

    hFea = {
        0: 'Feature_0',
        1: 'Feature_1',
        2: 'Feature_2',
        3: 'Feature_3',
        4: 'Feature_4',
        5: 'Feature_5',
        6: 'Feature_6',
        7: 'Feature_7',
        8: 'Feature_8',
        9: 'Feature_9',
        }

    for dIdx in range(10):
        plt.figure()
        plt.xlabel(hFea[dIdx])
        plt.hist(D0[dIdx, :], bins = 80, density = True, alpha = 0.4, label = 'Spoofed-fingerprint')
        plt.hist(D1[dIdx, :], bins = 80, density = True, alpha = 0.4, label = 'Authentic-fingerprint')
        
        
        plt.legend()
        plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
        plt.savefig('hist_%d.png' % dIdx)
    plt.show()
    
def plot_heatmap(D, L):

    D0 = D[:, L==0]
    D1 = D[:, L==1]
    

    hFea = {
        0: 'Feature_0',
        1: 'Feature_1',
        2: 'Feature_2',
        3: 'Feature_3',
        4: 'Feature_4',
        5: 'Feature_5',
        6: 'Feature_6',
        7: 'Feature_7',
        8: 'Feature_8',
        9: 'Feature_9',
        }
    # calculate Pearson correlation matrix
    corr_matrix = numpy.corrcoef(D1)
    
    fig, ax = plt.subplots(figsize=(12, 12))
    plt.imshow(corr_matrix, cmap='seismic')
    plt.colorbar()
    
    # set tick labels for x and y axes
    ax.set_xticks(numpy.arange(len(corr_matrix)))
    ax.set_yticks(numpy.arange(len(corr_matrix)))
    ax.set_xticklabels(numpy.arange(len(corr_matrix)))
    ax.set_yticklabels(numpy.arange(len(corr_matrix)))
    
    plt.title('Pearson Correlation Heatmap - \'Authentic fingerprints\' training set')    
        
        
    plt.legend()
    plt.tight_layout()
    plt.savefig('heatmap_training_set_authentic.png',dpi=300)
    plt.show()    

def plot_scatter(D, L):
    
    D0 = D[:, L==0]
    D1 = D[:, L==1]
    

    hFea = {
        0: 'Feature_0',
        1: 'Feature_1',
        2: 'Feature_2',
        3: 'Feature_3',
        4: 'Feature_4',
        5: 'Feature_5',
        6: 'Feature_6',
        7: 'Feature_7',
        8: 'Feature_8',
        9: 'Feature_9',
        }

    for dIdx1 in range(10):
        for dIdx2 in range(10):
            if dIdx1 == dIdx2:
                continue
            plt.figure()
            plt.xlabel(hFea[dIdx1])
            plt.ylabel(hFea[dIdx2])
            plt.scatter(D0[dIdx1, :], D0[dIdx2, :], label = 'Spoofed-fingerprint')
            plt.scatter(D1[dIdx1, :], D1[dIdx2, :], label = 'Authentic-fingerprint')
            
        
            plt.legend()
            plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
            plt.savefig('scatter_%d_%d.png' % (dIdx1, dIdx2))
        plt.show()

    
if __name__ == '__main__':

    # Change default font size - comment to use default values
    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    D, L = load('Train.txt')
    
    #plot_hist(D, L)
    #plot_scatter(D, L)
    plot_heatmap(D, L)

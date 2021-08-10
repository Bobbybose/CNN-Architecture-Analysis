#Analysis for three CNNs using different architectures
#Author: Bobby Bose

#Constants

#VGG16, GoogleNet, ResNet
MACSperFrame = [15500000000, 1430000000, 3900000000]
CNNNames = ["VGG16", "GoogleNet", "ResNet"]

#Global variables
clusteredTimeDrops = [145000000, 405000000, 1450000000, 5625000000]
distributedTimeDrops = [85000000, 145000000, 385000000, 1330000000, 5135000000]

class Architecture:
    def __init__(self, name, latency, energy):
        self.name = name
        self.MACLatency = latency
        self.MACEnergy = energy
        self.FPS = []
        self.totalEnergy = []
        self.totalFrames = []

def analysis(arch, maxParallelism, PDNType):
    framesPerSecond = []
    energyTot = []
    frameTotals = []
    i = 0

    for CNNMacs in MACSperFrame:
        print("Now starting", CNNNames[i])
        i += 1

        currentParallelism = maxParallelism
        PE = 32*8*2*maxParallelism

        totalFrames = 0
        totalTime = 0
        totalEnergy = 0

        frameLatency = ((arch.MACLatency*CNNMacs)*1e-9)/PE
        frameEnergy = arch.MACEnergy*CNNMacs*1e-12

        if(PDNType == 0):
            currTimeDrop = clusteredTimeDrops[0]
        else:
            currTimeDrop = distributedTimeDrops[0]
        
        timeDropPos = 0

        while currentParallelism >= 1:
            totalFrames += 1
            totalTime += frameLatency
            totalEnergy += frameLatency

            #If parallelism needs to drop
            if(totalTime >= currTimeDrop):
                
                #If we're at the end of the lifetime
                if(currentParallelism == 1):
                    framesPerSecond.append(totalFrames/totalTime)
                    energyTot.append(totalEnergy)
                    frameTotals.append(totalFrames)
                    break

                timeDropPos += 1                
                if(PDNType == 0):
                    currTimeDrop = clusteredTimeDrops[timeDropPos]
                else:
                    currTimeDrop = distributedTimeDrops[timeDropPos]


                currentParallelism /= 2
                PE = 32*8*2*currentParallelism
                frameLatency = ((arch.MACLatency*CNNMacs)*1e-9)/PE
                frameEnergy = arch.MACEnergy*CNNMacs*1e-12

    return framesPerSecond, energyTot, frameTotals


def main():
    Architectures = []

    Architectures.append(Architecture("DrAccClustered", 2940, 588))
    Architectures.append(Architecture("DrAccDistributed", 2940, 588))
    Architectures.append(Architecture("ELP2IMClustered", 3136, 448) )
    Architectures.append(Architecture("ELP2IMDistributed", 3136, 448))
    Architectures.append(Architecture("LAccClustered", 231, 150)) 
    Architectures.append(Architecture("LAccDistributed", 231, 150))

    typePDN=0
    for arch in Architectures:
        if(typePDN%2 == 0):
            maxPara = 8
        else:
            maxPara = 16

        arch.FPS, arch.totalEnergy, arch.totalFrames = analysis(arch, maxPara, typePDN)
        typePDN += 1

        print("{}:".format(arch.name))

        i = 0
        while i < 3:
            print("\t{}:".format(CNNNames[i]))  
            print (arch.FPS[i])
            print(arch.totalEnergy[i])
            print(arch.totalFrames[i])      


if __name__ == "__main__":
    main()

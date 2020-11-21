import os
import pandas as pd
from Pyfhel import Pyfhel, PyPtxt, PyCtxt
from datetime import datetime
import numpy as np
import sys
import time
import matplotlib.pyplot as plt


class homomorphic_encryption():
    def __init__(self, array):
        
        # Gerar chaves
        self.HE = Pyfhel()        
        self.HE.contextGen(p=65537) 
        self.HE.keyGen()
        
        # animacao de loading
        self.loading_animation = ['|', '/', '-', '\\']
        
        # array encriptado
        self.encrypt_data = self._encrypt_array(np.sort(np.array(array)))
                
        # tamanho do dados
        self.size = len(array)
        
        # estatisticas
        self.max = -1
        self.min = -1
        self.mean = -1
        self.sum = -1
        self.variance = -1
        
        # Arrays de tempos de execução
        self.time_sum = []
        self.time_max = []
        self.time_min = []
        self.time_variance = []
        self.time_normalize = []
        self.x_plot = list(range(10,self.size,50))
    
    ''' METODO REMOTO QUE CORRE DO LADO DO CLIENTE'''
    # Metodo que arredonda os dados encriptados
    def round_encryped(self,n,d):
        return self.HE.encryptFrac(round(self.HE.decryptFrac(n),d))

    # Este metodo percorre os dados raw e encripta-os
    def _encrypt_array(self,original_data):
        print("[LOAD ARRAY]\n")
        start_t = time.time()
        encrypt_data = []
        size = len(original_data)
        for i,elem in enumerate(original_data):
            sys.stdout.flush()
            sys.stdout.write("\rLoading "+self.loading_animation[i%4]+" ["+str(i+1)+"/"+str(size)+"]")
            encrypt_data.append(self.HE.encryptFrac(elem))
        sys.stdout.flush()
        sys.stdout.write("\rDone! - ["+str(size)+"/"+str(size)+"]")
        end_t = time.time()
        print("\nTime Elapsed: "+str(round(end_t - start_t,4))+" seconds!")
        return encrypt_data
        
    # Metodo para calcular a mediana
    def calculate_median(self):
        if self.size%2 == 1:
            return self.encrypt_data[self.size//2]
        else:
            aux = self.HE.encryptFrac(1/2)
            return (self.encrypt_data[self.size//2] + self.encrypt_data[self.size//2-1])*aux
    
    # Metodo para calcular a Q1
    def calculate_Q1(self):
        if self.size%4 == 1:
            return self.encrypt_data[self.size//4]
        else:
            aux = self.HE.encryptFrac(1/2)
            return (self.encrypt_data[self.size//4] + self.encrypt_data[self.size//4-1])*aux
    
    # Metodo para calcular a Q3
    def calculate_Q3(self):
        if self.size%4 == 1:
            return self.encrypt_data[self.size//4+self.size//2]
        else:
            aux = self.HE.encryptFrac(1/2)
            return (self.encrypt_data[self.size//4+self.size//2] + self.encrypt_data[self.size//4-1+self.size//2])*aux
    
    # Metodo para calcular a valor maximo
    def calculate_max(self):
        print("[CALCULATE MAX]")
        start_t = time.time()
        self.max = self.encrypt_data[self.size-1]
        '''
        Implemetacao antiga com complexidade O(n)
        self.max = self.HE.encryptFrac(-2163516721.0)
        for i,elem in enumerate(self.encrypt_data[:self.size]):
            sys.stdout.flush()
            sys.stdout.write("\rLoading "+self.loading_animation[i%4]+" ["+str(i+1)+"/"+str(self.size)+"]")
            if self.HE.decryptFrac(self.max - elem)<0:
                self.max = elem
        '''
        end_t = time.time()
        print("\nTime Elapsed: "+str(round(end_t - start_t,4))+" seconds!")
        return round(end_t - start_t,4)
    
    # Metodo para calcular o valor minimo
    def calculate_min(self):
        print("[CALCULATE MIN]")
        start_t = time.time()
        self.min = self.encrypt_data[0]
        ''' 
        Implemetacao antiga com complexidade O(n)
        self.min = self.HE.encryptFrac(635154214.0)
        for i,elem in enumerate(self.encrypt_data[:self.size]):
            sys.stdout.flush()
            sys.stdout.write("\rLoading "+self.loading_animation[i%4]+" ["+str(i+1)+"/"+str(self.size)+"]")
            if self.HE.decryptFrac(self.min - elem) > 0:
                self.min = elem
        '''
        end_t = time.time()
        print("\nTime Elapsed: "+str(round(end_t - start_t,4))+" seconds!")
        return round(end_t - start_t,4)
    
    # Metodo para calcular a soma do array
    def calculate_sum(self):
        print("[CALCULATE SUM]")
        start_t = time.time()
        tot = self.HE.encryptFrac(0.0)
        for i,elem in enumerate(self.encrypt_data[:self.size]):
            sys.stdout.flush()
            sys.stdout.write("\rLoading "+self.loading_animation[i%4]+" ["+str(i+1)+"/"+str(self.size)+"]")
            tot = tot + elem

        self.sum = self.round_encryped(tot,2)
        end_t = time.time()
        print("\nTime Elapsed: "+str(round(end_t - start_t,4))+" seconds!")
        return round(end_t - start_t,4)
    
    # Metodo para calcular a media
    def calculate_mean(self):
        print("[CALCULATE MEAN]")
        start_t = time.time()
        tot = self.sum
        size_encr = self.HE.encryptFrac(1.0/float(self.size))
        self.mean = tot*size_encr
        end_t = time.time()
        print("\nTime Elapsed: "+str(round(end_t - start_t,4))+" seconds!")
        return round(end_t - start_t,4)
    
    # Metodo para calcular a variancia // Que nao vai ser usado por causa dos arredondamentos
    def calculate_variance(self):
        print("[CALCULATE VARIANCE]")
        start_t = time.time()
        if self.mean == -1:
            self.calculate_mean()
        self.mean = self.round_encryped(self.mean,2)
        tot = self.HE.encryptFrac(0.0)
        for i,elem in enumerate(self.encrypt_data[:self.size]):
            sys.stdout.flush()
            sys.stdout.write("\rLoading "+self.loading_animation[i%4]+" ["+str(i+1)+"/"+str(self.size)+"]")
            diff = self.round_encryped(elem-self.mean,2)
            tot = self.round_encryped(tot + diff*diff,2)
        size_encr = self.round_encryped(self.HE.encryptFrac(1.0/float(self.size)),6)
        self.variance = self.round_encryped(tot*size_encr,2)
        end_t = time.time()
        print("\nTime Elapsed: "+str(round(end_t - start_t,4))+" seconds!")
        return round(end_t - start_t,4)

    def normalize_data(self):
        print("[NORMALIZE]")
        if self.max == -1:
            self.calculate_min()
        if self.min == -1:
            self.calculate_max()
        
        ''' Esta comparação é feita do lado do cliente, o servidor envia os valores e o cliente responde com True ou False '''
        if self.HE.decryptFrac(self.max) == self.HE.decryptFrac(self.min):
            return self.encrypt_data[:self.size]
        
        ''' Esta divisão é feita do lado do cliente '''
        divider = self.HE.encryptFrac(1.0/self.HE.decryptFrac(self.max - self.min))
        normalized_array = []
        for i,elem in enumerate(self.encrypt_data[:self.size]):
            sys.stdout.flush()
            sys.stdout.write("\rLoading "+self.loading_animation[i%4]+" ["+str(i+1)+"/"+str(self.size)+"]")
            normalized_array.append((elem-self.min)*divider)
        return normalized_array
    
    def count_time_normalize(self):
        start_t = time.time()
        self.normalize_data()
        end_t = time.time()
        print("\nTime Elapsed: "+str(round(end_t - start_t,4))+" seconds!")
        return round(end_t - start_t,4)

    # Metodo gerar os arrays com os tempos
    def generate_times(self):
        self.time_sum = []
        self.time_max = []
        self.time_min = []
        self.time_normalize = []
        self.time_variance = []
        act_size = self.size
        for i in range(10,self.size,50):
            print("CALCULATE FOR LEN - ",str(i))
            self.size = i
            self.time_sum.append(self.calculate_sum())
            self.time_max.append(self.calculate_max())
            self.time_min.append(self.calculate_min())
            self.time_variance.append(self.calculate_variance())
            self.time_normalize.append(self.count_time_normalize())
            
        self.size = act_size
        # Gera CSV
        pd.DataFrame(np.array([np.array(self.time_max),np.array(self.time_min),np.array(self.time_sum),np.array(self.time_variance),np.array(self.time_normalize)]).T, columns = ['Max',"Min",'Sum',"Variance","Norm"]).to_csv('Time_Analysis.csv') 
    
    # Metodo para dar plots
    def plot_times(self):
        
        fig = plt.figure()
        plt.scatter(self.x_plot,self.time_sum)
        plt.plot(self.x_plot,self.time_sum)
        plt.xlabel("Data's Size")
        plt.ylabel("Time in seconds")
        plt.title("Calculate Sum")
        fig.savefig('Plots/Calculate_Sum.jpg', bbox_inches='tight', dpi=150)
        plt.show()

        
        fig = plt.figure()
        plt.scatter(self.x_plot,self.time_max)
        plt.plot(self.x_plot,self.time_max)
        plt.xlabel("Data's Size")
        plt.ylabel("Time in seconds")
        plt.title("Find max value Data")
        fig.savefig('Plots/Calculate_Min.jpg', bbox_inches='tight', dpi=150)
        plt.show()
        
        fig = plt.figure()
        plt.scatter(self.x_plot,self.time_min)
        plt.plot(self.x_plot,self.time_min)
        plt.xlabel("Data's Size")
        plt.ylabel("Time in seconds")
        plt.title("Find min value Data")
        fig.savefig('Plots/Calculate_Max.jpg', bbox_inches='tight', dpi=150)
        plt.show()

        fig = plt.figure()
        plt.scatter(self.x_plot,self.time_variance)
        plt.plot(self.x_plot,self.time_variance)
        plt.xlabel("Data's Size")
        plt.ylabel("Time in seconds")
        plt.title("Calculate Variance Data")
        fig.savefig('Plots/Calculate_Variance.jpg', bbox_inches='tight', dpi=150)
        plt.show()
        
        fig = plt.figure()
        plt.scatter(self.x_plot,self.time_normalize)
        plt.plot(self.x_plot,self.time_normalize)
        plt.xlabel("Data's Size")
        plt.ylabel("Time in seconds")
        plt.title("Normalize Data")
        fig.savefig('Plots/Normalize_Data.jpg', bbox_inches='tight', dpi=150)
        plt.show()
        

        
    # Metodo que retorna as estatisticas todas extraidas
    ''' Este metodo está no lado do cliente, de modo a ser so ele a ter acesso aos valores estatisticos '''
    def get_all_statistics_decrypted(self):
        start_t = time.time()
        if(self.max == -1):
            print("Max not yet calculated, call method calculate_max() to get this statistic\n")
        else:
            print("MAX -> ",round(self.HE.decryptFrac(self.max),2),"\n")
        if(self.min == -1):
            print("Min not yet calculated, call method calculate_min() to get this statistic\n")
        else:
            print("MIN -> ",round(self.HE.decryptFrac(self.min),2),"\n")
        if(self.mean == -1):
            print("Mean not yet calculated, call method calculate_mean() to get this statistic\n")
        else:
            print("MEAN -> ",round(self.HE.decryptFrac(self.mean),2),"\n")
        if(self.sum == -1):
            print("Sum not yet calculated, call method calculate_sum() to get this statistic\n")
        else:
            print("SUM -> ",round(self.HE.decryptFrac(self.sum),2),"\n")
        
        if(self.variance == -1):
            print("Variance not yet calculated, call method calculate_variance() to get this statistic\n")
        else:
            print("VARIANCE -> ",round(self.HE.decryptFrac(self.variance),2),"\n")
        
        print("Median -> ",round(self.HE.decryptFrac(self.calculate_median()),2),"\n")
        
        print("First Quartile -> ",round(self.HE.decryptFrac(self.calculate_Q1()),2),"\n")
        
        print("Third Quartile -> ",round(self.HE.decryptFrac(self.calculate_Q3()),2),"\n")
        
        print("Inter-Quartile Range -> ",round(self.HE.decryptFrac(self.calculate_Q3() - self.calculate_Q1()),2),"\n")
        
        end_t = time.time()
        print("\nTime Elapsed: "+str(round(end_t - start_t,4))+" seconds!")



if __name__ == "__main__":
    covid_data = pd.read_csv("full_grouped.csv")
    encrypted_obj = homomorphic_encryption(covid_data["New cases"].tail(100))
    encrypted_obj.calculate_max()
    encrypted_obj.calculate_min()
    encrypted_obj.calculate_sum()
    encrypted_obj.calculate_mean()
    encrypted_obj.calculate_variance()
    encrypted_obj.get_all_statistics_decrypted()
    encrypted_obj.normalize_data()
    encrypted_obj.generate_times()
    encrypted_obj.plot_times()


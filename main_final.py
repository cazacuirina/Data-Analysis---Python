import numpy as np
from functii import *
import matplotlib.pyplot as plt
import seaborn as sb
from seaborn import heatmap
import statistics as st

#-------------------------------CITIRE date din fișiere CVS----------------------
nrTari,nrInd,linii,tari,indicatori,tabel,matrice=citireCsv("DSAddedValue.csv")
print(tabel,matrice)

nrTari,nrIndEmp,linii2,tari,indicatori_emp,tabel_emp,matrice_emp=citireCsv("DSEmployement.csv")
# print(tabel_emp,matrice_emp)


#------------------------------CALCUL MATRICEAL (NUMPY) & Salvare rezultate in CSV----------------
#           MINIM SI MAXIM
vect_maxime=MaxMin(matrice)[0]
vect_minime=MaxMin(matrice)[1]
max_indicator=list()
min_indicator=list()
for i in range(nrInd):
    max_indicator.append(tari[vect_maxime[i]])
    min_indicator.append(tari[vect_minime[i]])
# print(max_indicator,min_indicator)
indicatori_minmax = pd.DataFrame(data={
    "Sector": indicatori,
     "TaraMin": max_indicator,
    "TaraMax": min_indicator
})
indicatori_minmax.to_csv("1.MinMax.csv", index=False)
max_indicator=[v for v in zip(indicatori,max_indicator)]
min_indicator=[v for v in zip(indicatori,min_indicator)]
# for i in range(nrInd):
#     print("MIN: ",min_indicator[i],"MAX: ",max_indicator[i])

#           MEDIE, ABATERE STANDARD, COVARIANTA
indicatori_stat=[v for v in zip(tari,MedieStdCvar(matrice))]
# for i in range(len(tari)):
#     print(indicatori_stat[i])
medie=[row[0] for row in MedieStdCvar(matrice)]
stdErr=[row[1] for row in MedieStdCvar(matrice)]
Cvar=[row[2] for row in MedieStdCvar(matrice)]
indicatori_csv = pd.DataFrame(data={
    "tara": tari,
     "Medie": medie,
    "Std_err": stdErr,
    "Covarianta":Cvar
})
indicatori_csv.to_csv("2.MedieStdCvar.csv", index=False)

#           CORELATII
indVal=[str(x) + "_val" for x in indicatori]
indEmp=[str(x) + "_emp" for x in indicatori_emp]
corelatie,r_x,r_y,r_xy=calculCorrcoef(tari,indicatori,tabel,tabel_emp)
matriceCorrel(r_x, indVal, indVal, "3.CorelX_X.csv")
matriceCorrel(r_y, indEmp, indEmp, "3.CorelY_Y.csv")
matriceCorrel(r_xy, indVal, indEmp, "3.CorelX_Y.csv")



#------------------------------CALCUL TABELAR (PANDAS) & Salvare rezultate in CSV----------------
dfMatVal=DataFrameMatr(matrice,indicatori,tari)
# print(dfMatVal)

dfMatEmp=DataFrameMatr(matrice_emp,indicatori_emp,tari)
# print(dfMatEmp)

#                   STATISTICI
dfMatVal.describe().to_csv("4.StatisticiValAdaugata.csv")
dfMatEmp.describe().to_csv("4.StatisticiNrAng.csv")

#                   COMPARATIE TOTAL CU MEDIE
dfMatValMed=dfMatVal
dfMatValMed['Total'] = dfMatVal.apply(suma, axis=1)
dfMatValMed["ComparatieMedie"]=np.where(dfMatVal['Total']>((dfMatVal['Total'].sum())/len(dfMatVal['Total'])), True, False)
dfMatValMed .to_csv("5.TotalComparatieMedie.csv")

#                   SUMA CUMULATIVA
dfMatEmpCum=((dfMatEmp[indicatori_emp] / dfMatEmp[indicatori_emp].sum()) * 100)
dfMatEmpCum=dfMatEmpCum.assign(CumSum_AGR=dfMatEmpCum[indicatori_emp[0]].cumsum(axis=0),
                               CumSum_CONS=dfMatEmpCum[indicatori_emp[1]].cumsum(axis=0),
                               CumSum_INDUS=dfMatEmpCum[indicatori_emp[2]].cumsum(axis=0),
                               CumSum_MFG=dfMatEmpCum[indicatori_emp[3]].cumsum(axis=0),
                               CumSum_SERV=dfMatEmpCum[indicatori_emp[4]].cumsum(axis=0))
# print(dfMatEmp)
dfMatEmpCum.to_csv("6.CumSumEmp.csv")


#----------------------------------GRUPARE, AGREGARE, SUMARIZARE---------------------
dfVal=creareDataFrame(linii)
# print(dfVal)
# print(dfVal.describe())

dfEmp=creareDataFrame(linii2)
# print(dfEmp)
# print(dfEmp.describe())

#                   GOURP BY
dfVal.groupby(by="Tara").mean().to_csv("7.GrupareMedieTara.csv")
dfVal.groupby(by="Sector_activitate").sum().to_csv("7.GrupareSumaIndic.csv")

#                   FILTRARE SI SORTARE
dfValFilter=dfVal[dfVal['Valoare_Indicator'] > 10].sort_values(by='Valoare_Indicator')
dfValFilter.to_csv("8.ValFiltrate.csv")

#                   MERGE & AGGREGATE
rez=pd.merge(dfVal, dfEmp, on="Tara")
rez_agg=rez.groupby(by="Tara").agg(val_added_MIN=("Valoare_Indicator_x",np.min),val_added_MAX=("Valoare_Indicator_x",np.max),val_added_SUM=("Valoare_Indicator_x",np.sum),val_added_MEAN=("Valoare_Indicator_x",np.mean))
rez_agg.to_csv("9.MergedIndic.csv")

#                   JOIN & RANK
dfComb=dfMatEmp[0:nrTari].set_index('tara').join(dfMatVal[0:nrTari].iloc[:,0:nrInd+1].set_index('tara'), lsuffix="_emp", rsuffix="_val", how="inner")
dfComb['Ranks_emp']=dfComb.iloc[:,0:nrInd+1].sum(axis=1).rank(ascending=False)
dfComb['Ranks_val']=dfComb.iloc[:,nrInd+1:2*nrInd+1].sum(axis=1).rank(ascending=False)
dfComb.loc["Total_Indic"]=round(dfComb.sum(),5)
dfComb.to_csv("10.JoinedRanks.csv")


#-----------------------------------GRAFICE (Salvare PNG)-----------------------

#              GROUPED BAR CHART
plt.title('Valoarea adaugata pe sectoare de activitate', fontsize=14)
sb.set(style='white')
sb.barplot(x='Tara', y='Valoare_Indicator', hue='Sector_activitate', data=dfVal, width=1,palette=['orange', 'yellow','green','blue','violet'])
plt.xticks(rotation=45)
plt.savefig('11.GroupedBarChart.png')
# plt.show()

#              PIE CHART
plt.clf()
plt.title('Ponderea val_adaugata pe toate sectoarele de activitate', fontsize=14)
mat=matrice.T
data=list()
for v in tari:
    data.append(np.sum(tabel[v]))
# print(data)
colors = sb.color_palette('pastel')
plt.pie(data, labels=tari, colors = colors,autopct = '%.2f%%')
plt.savefig('12.PieChart.png')
# plt.show()

#              SCATTER PLOT
plt.clf()
plt.title('Angajati pe sectoare de activitate', fontsize=14)
plt.xlabel("TARI", size=14)
plt.ylabel("NR ANGAJATI", size=14)
scatter=plt.scatter(dfEmp["Tara"],dfEmp["Valoare_Indicator"],c=dfEmp.Sector_activitate.astype('category').cat.codes)
plt.legend(handles=scatter.legend_elements()[0],
           labels=indicatori,
           title="INDIC")
plt.savefig('13.ScatterPlot.png')
# plt.show()


#              HEAT MAP
plt.clf()
matMea=(dfMatEmp[indicatori_emp] / dfMatEmp[indicatori_emp].sum()) * 100
corel=CorelareColoaneEmpVal(dfMatVal,matMea,indicatori,indicatori_emp)
dfCorel=pd.DataFrame(corel)
plt.title('Corelatii valoare adaugata-numar angajati pe sectoare activitate', fontsize=14)
dataplot = sb.heatmap(dfCorel, cmap="winter", annot=True)
plt.savefig('14.HeatMap.png')
# plt.show()

"""Small serializable TF-IDF logistic classifier with held-out sigmoid calibration."""
import numpy as np
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from ml.reliable.preprocessing import normalize_sms,url_model_text

class Detector:
    def __init__(self,kind,C=1.0,class_weight=None):
        self.kind=kind;self.C=C;self.class_weight=class_weight;self.calibrator=None
    def texts(self,values):
        fn=url_model_text if self.kind=='url' else normalize_sms
        return [fn(v) for v in values]
    def fit(self,values,labels):
        char=TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=2,max_features=100000,sublinear_tf=True,dtype=np.float64)
        self.features=char if self.kind=='url' else FeatureUnion([('word',TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=100000,sublinear_tf=True)),('char',char)])
        x=self.features.fit_transform(self.texts(values))
        self.classifier=LogisticRegression(C=self.C,class_weight=self.class_weight,solver='liblinear',max_iter=1000,random_state=42)
        self.classifier.fit(x,labels)
        if (self.classifier.n_iter_>=1000).any(): raise RuntimeError('Classifier did not converge.')
        return self
    def raw_scores(self,values):
        return self.classifier.decision_function(self.features.transform(self.texts(values)))
    def calibrate(self,values,labels):
        self.calibrator=LogisticRegression(C=1.0,solver='lbfgs',random_state=42)
        self.calibrator.fit(self.raw_scores(values).reshape(-1,1),labels)
    def probability(self,values,calibrated=True):
        scores=self.raw_scores(values)
        if calibrated and self.calibrator is not None:
            return self.calibrator.predict_proba(scores.reshape(-1,1))[:,1]
        return 1/(1+np.exp(-np.clip(scores,-700,700)))
    def evidence(self,value):
        x=self.features.transform(self.texts([value])).tocsr()
        names=self.features.get_feature_names_out();weights=self.classifier.coef_[0]
        items=sorted(((int(i),float(v*weights[i])) for i,v in zip(x.indices,x.data)),key=lambda item:abs(item[1]),reverse=True)[:5]
        return [{'feature':str(names[i]),'contribution':round(v,4),'direction':'toward_threat' if v>0 else 'toward_benign'} for i,v in items]

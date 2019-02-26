# -*- coding: utf-8 -*-
from sklearn.svm import SVC
from zsutils import load_pkl, save_pkl, load_train_test, classify_analysis


if __name__ == '__main__':
    path = './data/train-test-0.4.npz'
    train_data, test_data, train_label, test_label = load_train_test(path)

    TEST = True
    model_save_path = './models/svc.pkl'
    if not TEST:
        model = SVC()
        model.fit(train_data, train_label)
        print('saving model...')
        save_pkl(model_save_path, model)
    else:
        print('reading model...')
        model = load_pkl(model_save_path)

    pred = model.predict(test_data)
    classify_analysis(test_label, pred)


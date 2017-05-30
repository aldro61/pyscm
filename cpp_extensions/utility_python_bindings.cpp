#include <Python.h>
#include <numpy/arrayobject.h>

#include <vector>
#include <iostream>

#include "best_utility.h"
#include "solver.h"

static PyObject *
find_max(PyObject *self, PyObject *args){
    double p;
    PyArrayObject *X, *y, *X_argsort_by_feature, *example_idx; //borrowed
    PyArrayObject *feature_weights = NULL;

    // Extract the argument values
    if(!PyArg_ParseTuple(args, "dO!O!O!O!|O!",
                         &p,
                         &PyArray_Type, &X,
                         &PyArray_Type, &y,
                         &PyArray_Type, &X_argsort_by_feature,
                         &PyArray_Type, &example_idx,
                         &PyArray_Type, &feature_weights)){
        return NULL;
    }

    // Check the type of the numpy arrays
    if(PyArray_TYPE(X) != PyArray_DOUBLE){
        PyErr_SetString(PyExc_TypeError,
                        "X must be numpy.ndarray type double");
        return NULL;
    }
    if(PyArray_TYPE(y) != PyArray_LONG){
        PyErr_SetString(PyExc_TypeError,
                        "y must be numpy.ndarray type int");
        return NULL;
    }
    if(PyArray_TYPE(X_argsort_by_feature) != PyArray_LONG){
        PyErr_SetString(PyExc_TypeError,
                        "X_argsort_by_feature must be numpy.ndarray type int");
        return NULL;
    }
    if(PyArray_TYPE(example_idx) != PyArray_LONG){
        PyErr_SetString(PyExc_TypeError,
                        "example_idx must be numpy.ndarray type int");
        return NULL;
    }
    if(feature_weights && PyArray_TYPE(feature_weights) != PyArray_DOUBLE){
        PyErr_SetString(PyExc_TypeError,
                        "feature_weights must be numpy.ndarray type double");
        return NULL;
    }

    // Check the number of dimensions of the numpy arrays
    if(PyArray_NDIM(X) != 2){
        PyErr_SetString(PyExc_TypeError,
                        "X must be a 2D numpy.ndarray");
        return NULL;
    }
    if(PyArray_NDIM(y) != 1){
        PyErr_SetString(PyExc_TypeError,
                        "y must be a 1D numpy.ndarray");
        return NULL;
    }
    if(PyArray_NDIM(X_argsort_by_feature) != 2){
        PyErr_SetString(PyExc_TypeError,
                        "X_argsort_by_feature must be a 2D numpy.ndarray");
        return NULL;
    }
    if(PyArray_NDIM(example_idx) != 1){
        PyErr_SetString(PyExc_TypeError,
                        "example_idx must be a 1D numpy.ndarray");
        return NULL;
    }
    if(feature_weights && PyArray_NDIM(example_idx) != 1){
        PyErr_SetString(PyExc_TypeError,
                        "feature_weights must be a 1D numpy.ndarray");
        return NULL;
    }

    // Check that the dimension sizes match
    npy_intp X_dim0 = PyArray_DIM(X, 0);
    npy_intp X_dim1 = PyArray_DIM(X, 1);
    npy_intp y_dim0 = PyArray_DIM(y, 0);
    npy_intp X_argsort_by_feature_dim0 = PyArray_DIM(X_argsort_by_feature, 0);
    npy_intp X_argsort_by_feature_dim1 = PyArray_DIM(X_argsort_by_feature, 1);
    npy_intp example_idx_dim0 = PyArray_DIM(example_idx, 0);
    npy_intp feature_weights_dim0 = 0;
    if(feature_weights){
        feature_weights_dim0 = PyArray_DIM(feature_weights, 0);
    }

    if(X_dim0 != y_dim0){
        PyErr_SetString(PyExc_TypeError,
                        "X and y must have the same number of rows");
        return NULL;
    }
    if(X_dim0 != X_argsort_by_feature_dim1){
        PyErr_SetString(PyExc_TypeError,
                        "X must have as many rows as X_argsort_by_feature has columns.");
        return NULL;
    }
    if(X_dim1 != X_argsort_by_feature_dim0){
        PyErr_SetString(PyExc_TypeError,
                        "X must have as many columns as X_argsort_by_feature has rows");
        return NULL;
    }
    if(feature_weights && feature_weights_dim0 != X_dim1){
        PyErr_SetString(PyExc_TypeError,
                        "feature_weights must have shape X.shape[1]");
        return NULL;
    }

    // Extract the data pointer from the number arrays
    double *X_data;
    long *y_data, *X_argsort_by_feature_data, *example_idx_data;
    X_data = (double*)PyArray_DATA(PyArray_GETCONTIGUOUS(X));
    y_data = (long*)PyArray_DATA(PyArray_GETCONTIGUOUS(y));
    X_argsort_by_feature_data = (long*)PyArray_DATA(PyArray_GETCONTIGUOUS(X_argsort_by_feature));
    example_idx_data = (long*)PyArray_DATA(PyArray_GETCONTIGUOUS(example_idx));

    double *feature_weights_data;
    if(feature_weights){
        feature_weights_data = (double*)PyArray_DATA(PyArray_GETCONTIGUOUS(feature_weights));
    }
    else{
        feature_weights_data = new double[X_dim1];
        std::fill_n(feature_weights_data, X_dim1, 1);
    }

    BestUtility best_solution(100);
    int status = find_max(p, X_data, y_data, X_argsort_by_feature_data, example_idx_data, feature_weights_data,
                          example_idx_dim0, X_dim0, X_dim1, best_solution);

    if(status != 0){
        PyErr_SetString(PyExc_TypeError,
                        "An error occurred in the solver");
        return NULL;
    }

    // Prepare variables for return

    double opti_utility = best_solution.best_utility;

    npy_intp dims[] = {best_solution.best_n_equiv};
    PyObject *opti_feat_idx = PyArray_SimpleNew(1, dims, PyArray_LONG);
    long *opti_feat_idx_data = (long*)PyArray_DATA(opti_feat_idx);

    PyObject *opti_thresholds = PyArray_SimpleNew(1, dims, PyArray_DOUBLE);
    double *opti_thresholds_data = (double*)PyArray_DATA(opti_thresholds);

    PyObject *opti_kinds = PyArray_SimpleNew(1, dims, PyArray_LONG);
    long *opti_kinds_data = (long*)PyArray_DATA(opti_kinds);

    PyObject *opti_N = PyArray_SimpleNew(1, dims, PyArray_LONG);
    long *opti_N_data = (long*)PyArray_DATA(opti_N);

    PyObject *opti_P_bar = PyArray_SimpleNew(1, dims, PyArray_LONG);
    long *opti_P_bar_data = (long*)PyArray_DATA(opti_P_bar);

    for(int i = 0; i < best_solution.best_n_equiv; i++){
        opti_feat_idx_data[i] = best_solution.best_feat_idx[i];
        opti_thresholds_data[i] = best_solution.best_feat_threshold[i];
        opti_kinds_data[i] = best_solution.best_feat_kind[i];
        opti_N_data[i] = best_solution.best_N[i];
        opti_P_bar_data[i] = best_solution.best_P_bar[i];
    }

    if (feature_weights){
        Py_DECREF(feature_weights);
    }
    else{
        delete [] feature_weights_data;
    }

    Py_DECREF(X);
    Py_DECREF(y);
    Py_DECREF(X_argsort_by_feature);
    Py_DECREF(example_idx);

    return Py_BuildValue("d,N,N,N,N,N",
                         opti_utility,
                         opti_feat_idx,
                         opti_thresholds,
                         opti_kinds,
                         opti_N,
                         opti_P_bar);
}


/***********************************************************************************************************************
 *                                                  MODULE DECLARATION
 **********************************************************************************************************************/
static PyMethodDef Methods[] = {
        {"find_max", find_max, METH_VARARGS,
                        "Find the split of maximum utility."},
        {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3

static struct PyModuleDef _scm_utility_module = {
    PyModuleDef_HEAD_INIT,
    "_scm_utility",
    NULL,
    -1,
    Methods
};

PyMODINIT_FUNC PyInit__scm_utility() {
    import_array();
    return PyModule_Create(&_scm_utility_module);
};

#else

PyMODINIT_FUNC
init_scm_utility
        (void){
    (void)Py_InitModule("_scm_utility", Methods);
    import_array();//necessary from numpy otherwise we crash with segfault
}

#endif
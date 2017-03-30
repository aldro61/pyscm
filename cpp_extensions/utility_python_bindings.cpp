#include <Python.h>
#include <numpy/arrayobject.h>
#include <iostream>

#include "utility.h"


/***********************************************************************************************************************
 *                                              MODULE GLOBAL VARIABLES
 **********************************************************************************************************************/
static PyObject *
find_max(PyObject *self, PyObject *args){
    PyArrayObject *X, *y, *X_argsort_by_feature, *example_idx; //borrowed
    PyArrayObject *feature_weights = NULL;

    // Extract the argument values
    if(!PyArg_ParseTuple(args, "O!O!O!O!|O!",
                         &PyArray_Type, &X,
                         &PyArray_Type, &y,
                         &PyArray_Type, &X_argsort_by_feature,
                         &PyArray_Type, &example_idx,
                         &PyArray_Type, &feature_weights)){
        return NULL;
    }
    std::cout << "FEATURE WEIGHTS: " << (feature_weights ? "yup" : "nope") << std::endl;

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
    npy_intp feature_weights_dim0 = PyArray_DIM(feature_weights, 0);
    if(X_dim0 != y_dim0){
        PyErr_SetString(PyExc_TypeError,
                        "X and y must have the same number of rows");
        return NULL;
    }
    if(X_dim0 != X_argsort_by_feature_dim0){
        PyErr_SetString(PyExc_TypeError,
                        "X and X_argsort_by_feature must have the same number of rows");
        return NULL;
    }
    if(X_dim1 != X_argsort_by_feature_dim1){
        PyErr_SetString(PyExc_TypeError,
                        "X and X_argsort_by_feature must have the same number of columns");
        return NULL;
    }
    if(y_dim0 != example_idx_dim0){
        PyErr_SetString(PyExc_TypeError,
                        "X and example_idx must have the same shape");
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

    if(feature_weights)
        std::cout << "It works with features!" << std::endl;
    else
        std::cout << "It works!" << std::endl;

    return Py_BuildValue("");
}


/***********************************************************************************************************************
 *                                                  MODULE DECLARATION
 **********************************************************************************************************************/
static PyMethodDef Methods[] = {
        {"find_max", find_max, METH_VARARGS,
                        "Find the split of maximum utility."},
        {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
init_scm_utility
        (void){
    (void)Py_InitModule("_scm_utility", Methods);
    import_array();//necessary from numpy otherwise we crash with segfault
}
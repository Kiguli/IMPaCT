#include "IMDP.h"

/// Custom PDF function, change this to the PDF function desired that will be integrated over with Monte Carlo integration
// double customPDF(double *x, size_t dim, void *params)
// {
//     //custom PDF parameters that are passed in (not all need to be used)
//     customParams *p = reinterpret_cast<customParams*>(params);
//     vec mean = p-> mean;
//     vec state_start = p->state_start;
//     function<vec(const vec&,const vec&)> dynamics2 = p-> dynamics2;
//     vec input = p-> input;
//     vec lb = p-> lb;
//     vec ub = p-> ub;
//     vec eta = p-> eta;
//     /* Other parameters of struct unused in the ex_custom_distribution example:*/
//     //function<vec(const vec&)> dynamics1 = p->dynamics1;
//     //function<vec(const vec&,const vec&)> dynamics3 = p-> dynamics3;
//     //vec disturb = p-> disturb;
//
//     //multivariate normal PDF example:
//     double cov_det = 0.5625;
//     mat inv_cov = {{1.3333, 0},{0, 1.3333}};
//     double norm = 1.0 / (pow(2 * M_PI, dim / 2.0) * sqrt(cov_det));
//
//     double exponent = 0.0;
//     for (size_t i = 0; i < dim; ++i) {
//         for (size_t j = 0; j < dim; ++j) {
//             exponent -= 0.5 * (x[i] - mean[i]) * (x[j] - mean[j]) * inv_cov(i,j);
//         }
//     }
//     return norm * exp(exponent);
// }

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "dl.h"

#define IN_SIZE 29
#define AFFINE_LAYER 3
#define AFFINE_SIZE1 IN_SIZE
#define AFFINE_SIZE2 8
#define THRESHOLD 0.5
#define LEARN 10000
#define LAMBDA 0.0
#define REG 1

int main (int argc, char** argv) {

  if(argv[1]=='\0') {
    printf("NN.out [sel] [ratio] [data] [weight]\n");
    return 0;
  }

  double* in;
  double** x;
  double** y;
  double** x_error;
  double** y_error;
  double*** w;
  double* tag;
  double* softmaxerror;
  int sel=atoi(argv[1]);
  double ratio=atof(argv[2]);
  int bit=0;
  int noerror=0;
  FILE* fp;
  FILE* fp2;
  char line[1000]={'\0'};
  int count=0;
  int bigerror=0;
  int learn = 0;
  double th=THRESHOLD;
  double maxerror=1;
  double maxdiff=0;
  int rep = 0 ;
  double ave = 0 ;
  double maxprop=0;
  double ignore=0;

  /* input layer */
  in=(double*)malloc(sizeof(double) * IN_SIZE);

  /*  declare */
  fp=fopen(argv[3],"r");
  fp2=fopen(argv[4],"w");
  x=(double**)malloc(sizeof(double*) * AFFINE_LAYER);
  y=(double**)malloc(sizeof(double*) * AFFINE_LAYER);
  for (int i=0; i < AFFINE_LAYER; i++){
    x[i]=(double*)malloc(sizeof(double) * AFFINE_SIZE1);
    y[i]=(double*)malloc(sizeof(double) * AFFINE_SIZE1);
  }

  w=(double***)malloc(sizeof(double**) * AFFINE_LAYER);
  for(int i=0;i<AFFINE_LAYER;i++){
    w[i]=(double**)malloc(sizeof(double*) * (AFFINE_SIZE1+1));
    if(i==AFFINE_LAYER-1){
      for(int j=0; j<AFFINE_SIZE1+1;j++){
        w[i][j]=(double*)malloc(sizeof(double) * AFFINE_SIZE2);
      }
      weight_init(w[i],AFFINE_SIZE1,AFFINE_SIZE2);
    }
    else {
      for(int j=0; j<AFFINE_SIZE1+1;j++){
        w[i][j]=(double*)malloc(sizeof(double) * AFFINE_SIZE1);
      }
      weight_init(w[i],AFFINE_SIZE1,AFFINE_SIZE1);
    }
  }

  tag=(double*)malloc(sizeof(double)*AFFINE_SIZE2);
  softmaxerror=(double*)malloc(sizeof(double)*AFFINE_SIZE2);

  x_error=(double**)malloc(sizeof(double*) * AFFINE_LAYER);
  y_error=(double**)malloc(sizeof(double*) * AFFINE_LAYER);
  for (int i=0; i < AFFINE_LAYER; i++){
    x_error[i]=(double*)malloc(sizeof(double) * AFFINE_SIZE1);
    y_error[i]=(double*)malloc(sizeof(double) * AFFINE_SIZE1);
  }

  while(fgets(line,999,fp)){
    ratio=atof(argv[2]);
    count++;
    noerror=1;
    rep = 1;
    maxprop=0;
    for (int i=0; i<=IN_SIZE; i++) {
      if(i==0) {
        in[i]=atof(strtok(line,","));
        while (in[i] > 1) {
          in[i] = in[i] / 10;
        }
      }
      else if(i==IN_SIZE){
        bit=atof(strtok(NULL,","));
      }
      else {
        in[i]=atof(strtok(NULL,","));
        while (in[i] > 1) {
          in[i] = in[i] / 10;
        }
      }
    }

    /* full connection layer*/
    for(int j=0;j<AFFINE_SIZE1;j++){
      x[0][j]=in[j];
    }
  REP:
    for (int i = 0 ; i < AFFINE_LAYER; i++) {
      if (i==AFFINE_LAYER-1){
        affine (x[AFFINE_LAYER-1],w[AFFINE_LAYER-1],y[AFFINE_LAYER-1],AFFINE_SIZE1,AFFINE_SIZE2,sel);
      }
      else {
        affine (x[i],w[i],y[i],AFFINE_SIZE1,AFFINE_SIZE1,sel);
        for(int j=0;j<AFFINE_SIZE1;j++){
          x[i+1][j]=y[i][j];
        }
      }
    }

    bigerror=0;
    make_tag(tag,AFFINE_SIZE2,bit);
    softmaxwithloss(y[AFFINE_LAYER-1], tag, softmaxerror, AFFINE_SIZE2);

    learn++;

    for(int i=0; i<AFFINE_SIZE2; i++) {
      if (softmaxerror[i] != 0){
        noerror=0;
      }
      if (softmaxerror[i] > th || softmaxerror[i] < -1 * th){
        bigerror=1;
      }
      if (learn == LEARN) {
        printf("%f ",softmaxerror[i]);
      }

      for (int i = 0; i < AFFINE_SIZE2; i++) {
        if (maxprop < softmaxerror[i]) {
          maxprop = softmaxerror[i];
        }
      }

      /* ignoring data  */
      ignore=fabs(ave) + fabs(th);
      if (ignore > 1) {
        ignore = 1;
      }
      if (count > 1 && fabs(maxprop) > ignore) {
        bigerror=2;
      }
      ave = (ave * (count-1) + fabs(maxprop)) / count;


      /* ratio renew */
      if (y_error[AFFINE_LAYER-1][i] > softmaxerror[i]) {
        if (maxerror < (y_error[AFFINE_LAYER-1][i] - softmaxerror[i])) {
          maxdiff=y_error[AFFINE_LAYER-1][i] - softmaxerror[i] -maxerror;
          maxerror = y_error[AFFINE_LAYER-1][i] - softmaxerror[i];
        }
      }
      else {
        if (maxerror < (softmaxerror[i] - y_error[AFFINE_LAYER-1][i])) {
          maxdiff=y_error[AFFINE_LAYER-1][i] - softmaxerror[i] -maxerror;
          maxerror = softmaxerror[i] - y_error[AFFINE_LAYER-1][i];
        }
      }

      y_error[AFFINE_LAYER-1][i]=softmaxerror[i];
    }
    if (learn == LEARN) {
      printf("%f %f %d %d %f\n",th,ratio,bigerror,count,ignore);

      learn=0;
    }

    /* back propagation layer */
    if (noerror==1){
      //end
      break;
    }
    if (bigerror==2){
      continue;
    }
    else {
      for (int i = AFFINE_LAYER - 1 ; i >= 0; i--) {
        if (i==0){
          backaffine (x[i],w[i],y[i],AFFINE_SIZE1,AFFINE_SIZE1,sel,x_error[i],y_error[i],ratio);
        }
        else {
          backaffine (x[i],w[i],y[i],AFFINE_SIZE1,AFFINE_SIZE1,sel,x_error[i],y_error[i],ratio);
          for(int j=0;j<AFFINE_SIZE1;j++){
            y_error[i-1][j]=x_error[i][j];
          }
        }
      }
    }
    /* rep , threshold renew, ratio renew */
    if(bigerror!=0){
      ratio = ratio * (1 + fabs(maxerror - th)/ th / exp(rep));
      th = th + fabs(maxerror / ratio);
      if (th > 1) {
        th = 1;
      }
      rep ++;
      goto REP;
    }
    else {
      //ratio = ratio * fabs(maxerror / th) ;
      th = th - fabs(maxerror / ratio) ;
      if (th < THRESHOLD) {
        th = THRESHOLD;
      }
    }
  }

  //end
  for(int i=0;i<AFFINE_LAYER;i++){
    fprintf(fp2,"weight layer%d : \n",i);
    for(int j=0; j<AFFINE_SIZE1+1;j++){
      if (i==AFFINE_LAYER-1){
        for (int k=0; k<AFFINE_SIZE2;k++){
          fprintf(fp2,"%f ",w[i][j][k]);
        }
      }
      else{
        for (int k=0; k<AFFINE_SIZE1;k++){
          fprintf(fp2,"%f ",w[i][j][k]);
        }
      }
      fprintf(fp2,"\n");
    }
    fprintf(fp2,"\n");
  }
  fclose(fp);
  fclose(fp2);

  free(in);
  for (int i=0; i < AFFINE_LAYER; i++){
   free(x[i]);
   free(y[i]);
  }
  free(x);
  free(y);
  for (int i=0; i < AFFINE_LAYER; i++){
    free(x_error[i]);
    free(y_error[i]);
  }
  free(x_error);
  free(y_error);
  for(int i=0;i<AFFINE_LAYER;i++){
    for(int j=0; j<AFFINE_SIZE1+1;j++){
          free(w[i][j]);
    }
    free(w[i]);
  }
  free(w);
  free(tag);
  free(softmaxerror);
  return 0;
}

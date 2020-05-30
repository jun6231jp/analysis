#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

double SIZE=1;
double GAIN=0.1;

/* Note
Deep Learningは多層にするほど表現力があがる。多層化するためには活性化関数が必要。
活性化関数がなければ多層化したとしても1層で表現できてしまう。
https://blog.apar.jp/deep-learning/12216/

多層化が進むと学習が収束しない問題や学習精度が向上しない問題があったため学習テクニックが使われる。
3層以上：ReLu、DropOut、係数初期値
10層以上：Batch Normalizationなどの正規化
20層以上：Residual Networkなどのスキップコネクション、AuxitiallyLoss
https://www.hellocybernetics.tech/entry/2017/01/31/041148
https://www.youtube.com/watch?v=X2KWO1UPqxk&list=PLg1wtJlhfh23pjdFv4p8kOBYyTRvzseZ3&index=6
https://www.youtube.com/watch?v=O3qm6qZooP0&list=PLg1wtJlhfh23pjdFv4p8kOBYyTRvzseZ3&index=4
https://www.youtube.com/watch?v=r8bbe273vEs&list=PLg1wtJlhfh23pjdFv4p8kOBYyTRvzseZ3&index=7

ゼロパディング（zero padding）
・端のデータに対する畳み込み回数が増えるので端の特徴も考慮されるようになる
・畳み込み演算の回数が増えるのでパラメーターの更新が多く実行される
・カーネルのサイズや、層の数を調整できる

Pooling層はConvolutoin層の後に適用される。入力データをより扱いやすい形に変形するために、情報を圧縮し、down samplingする。
https://deepage.net/deep_learning/2016/11/07/convolutional_neural_network.html#畳み込みとは

・Convolutional Layer: 特徴量の畳み込みを行う層
・Pooling Layer: レイヤの縮小を行い、扱いやすくするための層
・Fully Connected Layer: 特徴量から、最終的な判定を行う層
https://qiita.com/icoxfog417/items/5fd55fad152231d706c2

ディープラーニングのネットワークの出力の最後にSoftmaxを置く。出力結果を確率に変換する。
http://hiro2o2.hatenablog.jp/entry/2016/07/21/013805

誤差逆伝播重ね合わせ
http://www.sist.ac.jp/~kanakubo/research/neuro/backpropagation.html
http://ipr20.cs.ehime-u.ac.jp/column/neural/chapter6.html

畳み込み層の逆伝播
http://www.thothchildren.com/chapter/5c4efaca41f88f26724e9a78
https://urusulambda.wordpress.com/2018/12/24/%E3%82%82%E3%81%86%E4%B8%80%E5%BA%A6%E3%81%BE%E3%81%98%E3%82%81%E3%81%ABconvolution%E3%83%AC%E3%82%A4%E3%83%A4%E7%95%B3%E3%81%BF%E8%BE%BC%E3%81%BF%E5%B1%A4%E3%81%AE%E5%9F%BA%E7%A4%8E%E3%82%92/
Max-Pooling逆伝播
https://www.tcom242242.net/entry/2017/04/23/213242/

オンライン学習：データごとにパラメータ更新。大きいデータセットに有効。
バッチ学習：データ全体を使ってパラメータ更新。小さいデータセットに有効。
ミニバッチ学習：データセットを分割しパラメータ更新。学習勾配が安定。

*/

/*活性化関数*/
double sigmoid(double gain, double x) {
  return 1.0 / (1.0 + exp(-gain * x));
}

double ReLu (double slope, double x) {
  if (x > 0) {
    return x * slope;
  }
  else {
    return 0;
  }
}

double SigLu (double gain, double slope, double intercept, double x) {
  return (1.0 / (1.0 + exp(-gain * x))) * (intercept + slope * x);
}

double Activation (double gain, double slope, double x, int sel) {
  if (sel == 0) {
    return sigmoid(gain,x);
  }
  else if (sel == 1) {
    return ReLu(slope,x);
  }
  else if (sel == 2) {
    return SigLu(gain,slope,1,x);
  }
}

/*forward*/
void zeropadding (double** in, double** out, int xsize, int ysize, int filtersize) {
  for (int i = 0 ; i < xsize + 2*filtersize -2 ; i++) {
    for (int j = 0 ; j < ysize + 2*filtersize -2 ; j ++) {
      out[i][j] = 0 ;
    }
  }
  for (int i = 0 ; i < xsize ; i++) {
    for (int j = 0 ; j < ysize ; j ++) {
      out[i+filtersize-1][j+filtersize-1]=in[i][j];
    }
  }
}

void convolution(double** in , double** filter, double** out, int xsize, int ysize, int filtersize){
  for (int i = 0 ; i < xsize - filtersize + 1 ; i++) {
    for (int j = 0 ; j < ysize - filtersize + 1 ; j++) {
      out[i][j] = 0;
    }
  }
  for (int i = 0 ; i < xsize - filtersize + 1; i++) {
    for (int j = 0 ; j < ysize - filtersize + 1 ; j++) {
      for (int k = 0 ; k < filtersize; k++) {
        for (int l = 0 ; l < filtersize; l++) {
          out[i][j] += in[i+l][j+k] * filter[l][k];
        }
      }
    }
  }
}

void maxpooling(double** in, double** out, int xsize, int ysize, int xfiltersize, int yfiltersize){
  for (int i = 0 ; i < xsize - xfiltersize + 1 ; i++) {
    for (int j = 0 ; j < ysize - yfiltersize + 1 ; j++) {
      out[i][j] = 0;
    }
  }
  for (int i = 0 ; i < xsize - xfiltersize + 1; i++) {
    for (int j = 0 ; j < ysize - yfiltersize + 1 ; j++) {
      for (int k = 0 ; k < xfiltersize; k++) {
        for (int l = 0 ; l < yfiltersize; l++) {
          if (out[i][j] < in[i+k][j+l]) {
            out[i][j] = in[i+k][j+l];
          }
        }
      }
    }
  }
}

void affine (double* x, double** w, double* y, int sizex, int sizey, int sel) {
  double* sum;
  sum=(double*)malloc(sizeof(double) * sizey);
  for(int i=0 ; i < sizey; i++) {
    sum[i]=0;
  }
  for (int j=0; j<sizey; j++) {
    sum[j]=w[0][j];
    for (int i=1; i < sizex + 1; i++) {
      sum[j] += x[i-1] * w[i][j];
    }
    y[j] = Activation(GAIN,1/SIZE,sum[j],sel);
  }
  free(sum);
}

void softmaxwithloss(double* in, double* tag, double * error, int size){
  double sum = 0;
  for (int i = 0 ; i < size; i++) {
    sum += in[i] ;
  }
  for (int i = 0 ; i < size; i++) {
    if (sum != 0) {
      error[i] = (in[i] / sum) - tag[i];
    }
    else {
      error[i] = in[i] - tag[i];
    }
  }
}

double** skipConnection(){

}

/* backward*/
double backSig (double x, double y, double error) {
  return error * y * (1 - y) * GAIN;
}

double backReLu(double x, double y, double error) {
  if (x > 0) {
    return error / SIZE;
  }
  else {
    return 0;
  }
}

double backSigLu (double x, double y, double error) {
  return error * (1/SIZE * (1 + 1 / exp(x)) - (y /exp(x)) );
  //return error * y * y * ( (0.01 - 1) / exp(x) + 0.01 * (1 - x / exp(x)) ) / (1 + x) / (1 + x);
}

double backActivation (double x, double y, double error, double ratio ,double sel) {
  if (sel == 0) {
    return ratio * backSig(x,y,error);
  }
  else if (sel == 1) {
    return ratio * backReLu(x,y,error);
  }
  else if (sel == 2) {
    return ratio * backSigLu(x,y,error);
  }
}

void backaffine (double* x, double** w, double* y, int sizex, int sizey, int sel, double* errorx, double* errory, double ratio){
  double* sum;
  double* backsum;

  sum=(double*)malloc(sizeof(double) * sizey);
  for(int i=0 ; i < sizey; i++) {
    sum[i]=0;
  }
 for (int j=0; j<sizey; j++) {
    sum[j]=w[0][j];
    for (int i=1; i < sizex + 1; i++) {
      sum[j] += x[i-1] * w[i][j];
    }
  }
  backsum=(double*)malloc(sizeof(double) * sizex);
  for(int i=0 ; i < sizex ; i++) {
    backsum[i]=0;
  }
  for (int i=1; i<sizex+1; i++) {
    for (int j=0; j < sizey; j++) {
      backsum[i-1] += w[i][j] * backActivation(sum[j],y[j],errory[j],ratio,sel);
    }
    errorx[i-1]=backsum[i-1];
  }
  for (int i=0; i<sizex + 1; i++) {
    for (int j=0; j < sizey; j++) {
      if (i==0){
        w[i][j] -= backActivation(sum[j],y[j],errory[j],ratio,sel);
      }
      else {
        w[i][j] -= backActivation(sum[j],y[j],errory[j],ratio,sel) * x[i-1];
      }
    }
  }
  free(sum);
  free(backsum);
}

void backconv(double** in , double** filter, double** out, int xsize, int ysize, int filtersize, double** error , double ratio){
  for (int k=0; k < filtersize; k++){
    for (int l=0; l < filtersize; l++){
      for (int i=0; i < xsize-filtersize+1; i++){
        for (int j=0; j < ysize-filtersize+1; j++){
          filter[k][l] -= ratio * error[i][j] * in[i+k][j+l] ;
        }
      }
    }
  }
}

void backpool(double** in , double** out, int xsize, int ysize, int xfiltersize , int yfiltersize, double** error, double ratio){
  for (int i=0; i < xsize-xfiltersize+1; i++){
    for (int j=0; j < ysize-yfiltersize+1; j++){
      for (int k=0; k < xfiltersize; k++){
        for (int l=0; l < yfiltersize; l++){
          if (in[i+k][j+l] == out[i][j]){
            in[i+k][j+l] -= ratio * error[k][l];
          }
        }
      }
    }
  }
}

void filter_init(double** filter, int filter_size){
  srand(1);
  for (int i = 0; i < filter_size ; i++) {
    for (int j = 0 ; j < filter_size ; j++) {
      filter[i][j]=(double)rand() / (double)RAND_MAX;
    }
  }
}

void weight_init(double**w,int in_size,int out_size){
  srand(1);
  for (int i = 0; i < (in_size + 1) ; i++) {
    for (int j = 0 ; j < out_size ; j++) {
      w[i][j]=(double)rand() / (double)RAND_MAX;
    }
  }
}

void weight_set(double**w,double** wt,int in_size,int out_size){
  for (int i = 0; i < (in_size + 1) ; i++) {
    for (int j = 0 ; j < out_size ; j++) {
      w[i][j]=wt[i][j];
    }
  }
}

void make_tag(double* tag,int size,int bit) {
  for (int i=0;i<size;i++) {
    tag[i]=0;
  }
  tag[bit - 1]=1;
}


double regularization(int lambda, double p, double* x, int size) {

  double sum=0;

  for (int i=0; i < size ; i++) {
    sum += pow(x[i],p);
  }
  sum=lambda * pow(1/p,sum);

  return sum;

}

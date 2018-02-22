// neuralplex is distributed under BSD license reproduced below.
//
// Copyright (c) 2015 Gregory "f3z0" Ray, f3z0@fezo.com
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//  * Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//  * Neither the name of the psutil authors nor the names of its contributors
//    may be used to endorse or promote products derived from this software without
//    specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
// ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <iostream>
#include <stdlib.h>
#include <cmath>
#include <unistd.h>
#include <sys/time.h>
#include "neural_net_constants.h"
#include "neural_net.h"
#include <mysql.h>
#include "rapidjson/filestream.h"
#include "rapidjson/prettywriter.h"

float TanhScaled(float x){
  return 1.7159*tanh(0.66666667*x);
}

float TanhScaledPrime(float x) {
  return 0.66666667/1.7159*(1.7159+tanh(x))*(1.7159-tanh(x));
}

float Sigmoid(float x) {
  return 1/(1+pow(exp(1.0), -x));
}

float SigmoidPrime(float x) {
  return Sigmoid(x) * (1.0-Sigmoid(x));
}

int main () {
  MYSQL *conn;
  MYSQL_RES *res_good;
  MYSQL_RES *res_bad;
  MYSQL_RES *res_all;
  MYSQL_ROW row;
  unsigned int n_field_max_length = 50;
  unsigned int n_hidden = 150;
  unsigned int n_output = 1;
  unsigned int n_positive_training_rows = 300;
  unsigned int n_negative_training_rows = 300;
  std::string good_user_query = ("SELECT * FROM (SELECT LCASE(email), LCASE(city), dateOfBirth, LCASE(billingAddress), phone, LCASE(firstName), LCASE(lastName), LCASE(zipcode), \
                                                billingAddress \
                                                from (SELECT sum(amount) as total, email, city, dateOfBirth, billingAddress, phone, firstName, lastName, zipcode FROM transaction_receipts JOIN User ON User.id = userId JOIN UserInformation ON information_fk = UserInformation.id WHERE productId = 'Slotser Deposit' AND (transaction_receipts.currency = 'GBP' OR transaction_receipts.currency = 'EUR') AND User.status = 'ENABLED' AND userType = 'USER' AND CHAR_LENGTH(billingAddress) > 10 GROUP BY userId) b where total >= 25.0) DerivedA \
                                        ORDER BY RAND() \
                                        LIMIT " + std::to_string(n_positive_training_rows));
  std::string bad_user_query = "SELECT \
                                                LCASE(email), \
                                                LCASE(city), \
                                                LCASE(billingAddress), \
                                                phone, \
                                                LCASE(firstName), \
                                                LCASE(lastName), \
                                                LCASE(zipcode), \
                                                billingAddress \
                                                from User \
                                                JOIN UserInformation ON information_fk = UserInformation.id \
                                                AND status = \"CLOSED\" \
                                                AND userType = \"USER\" \
                                                AND dateOfBirth IS NOT NULL \
                                                ORDER BY RAND() \
                                                LIMIT " + std::to_string(n_negative_training_rows);
  std::string all_user_query = "SELECT \
                                              LCASE(email), \
                                              LCASE(city), \
                                              LCASE(billingAddress), \
                                              phone, \
                                              LCASE(firstName), \
                                              LCASE(lastName), \
                                              LCASE(zipcode), \
                                                billingAddress, \
                                              User.id, \
                                              status \
                                              FROM User \
                                              JOIN UserInformation ON information_fk = UserInformation.id \
                                              WHERE  (status = \"ENABLED\" \
                                              OR status = \"CLOSED\") \
                                              AND userType = \"USER\" \
                                              AND dateOfBirth IS NOT NULL;";
  char const *server = "127.0.0.1";
  char const *user = "un";
  char const *password = "ps" ;
  char const *database = "db";
  conn = mysql_init(NULL);
  if (!mysql_real_connect(conn, server,
       user, password, database, 3307, NULL, 0)) {
    fprintf(stderr, "%s\n", mysql_error(conn));
    exit(1);
  }
  if (mysql_query(conn, good_user_query.c_str())) {
    fprintf(stderr, "%s\n", mysql_error(conn));
    exit(1);
  }
  res_good = mysql_store_result(conn);
  if (mysql_query(conn, bad_user_query.c_str())) {
    fprintf(stderr, "%s\n", mysql_error(conn));
    exit(1);
  }
  res_bad = mysql_store_result(conn);
  unsigned int n_fields = mysql_field_count(conn);
  unsigned int n_good_rows = mysql_num_rows(res_good);
  unsigned int n_bad_rows = mysql_num_rows(res_bad);
  unsigned int batch_size = n_good_rows+n_bad_rows;
  float training_data_arr[n_good_rows+n_bad_rows][n_fields*n_field_max_length+1];
  unsigned int k = 0;
  while ((row = mysql_fetch_row(res_good)) != NULL) {
    for (unsigned int j = 0; j < n_fields; j++) {
      for (unsigned int i = 0; i < n_field_max_length; i++) {
        float c = 0;
        if (row[j] && i < strlen(row[j])) c = (float)row[j][i];
        training_data_arr[k][j*n_field_max_length+i] = c;
      }
    }
    training_data_arr[k][n_field_max_length*n_fields] = 1.0f;
   // training_data_arr[k][n_field_max_length*n_fields+1] = -1.0f;
    k++;
  }
  while ((row = mysql_fetch_row(res_bad)) != NULL) {
    for (unsigned int j = 0; j < n_fields; j++) {
      for (unsigned int i = 0; i < n_field_max_length; i++) {
        //std::cout << "FF: char: " << row[j][i]  << std::endl;
        int c = 0;
        if (row[j] && i < strlen(row[j])) c = (int)row[j][i];
        training_data_arr[k][j*n_field_max_length+i] = c;
        //std::cout << "FF: char: " << row[j][i] << " int: " << (int)row[j][i] << " float1: " << (float)(int)row[j][i] << " float2: " << training_data_arr[k][j*n_field_max_length+i] << std::endl;
      }
    }
    training_data_arr[k][n_field_max_length*n_fields] = 0.0f;
  //  training_data_arr[k][n_field_max_length*n_fields+1] = 1.0f;
    k++;
  }
  mysql_free_result(res_good);
  mysql_free_result(res_bad);
  if (mysql_query(conn, all_user_query.c_str())) {
    fprintf(stderr, "%s\n", mysql_error(conn));
    exit(1);
  }
  res_all = mysql_store_result(conn);
  unsigned int n_all_rows = mysql_num_rows(res_all);
  std::vector<std::string> user_ids;
  std::vector<std::string> emails;
  std::vector<std::string> statuses;

  std::vector < std::vector<float> > test_data_arr;
  //test_data_arr = (float*) malloc(n_all_rows * n_fields * n_field_max_length * sizeof(float));

  for (unsigned int i = 0; i < n_all_rows; i++) {
    std::vector<float> init_data;
    for (unsigned int j = 0; j < (n_fields*n_field_max_length); j++) {
      init_data.push_back(0.0f);
    }
    test_data_arr.push_back(init_data);
  }
  k = 0;

  while ((row = mysql_fetch_row(res_all)) != NULL) {
    for (unsigned int j = 0; j < n_fields; j++) {
      for (unsigned int i = 0; i < n_field_max_length; i++) {
        float c = 0;

        if (row[j] && i < strlen(row[j])) c = (float)(int)row[j][i];
        test_data_arr[k][j*n_field_max_length+i] = c;
     // std::cout << "strlen(row[j]): " << strlen(row[j]) << " i: " << i <<  "FF: char: " << c << " int: " << (int)row[j][i] << " float1: " << (float)(int)row[j][i] << " float2: " << test_data_arr[k][j*n_field_max_length+i] << std::endl;
      }
    }
    user_ids.push_back(row[n_fields]);
    emails.push_back(row[0]);
    statuses.push_back(row[n_fields+1]);
    k++;
  }

  mysql_free_result(res_all);
  //std::random_shuffle(training_data_arr[0], training_data_arr[batch_size-1]);
  //for (int i = 0; i < batch_size; i++) { for (int j = 0; j < n_fields*n_field_max_length+1; j++) { std::cout << training_data_arr[i][j] << ","; }; std::cout << std::endl;  }
  unsigned int n_input = n_fields*n_field_max_length;
  bool did_converge = false;
  long long elapsed_time  = 0;
  struct timeval start, end;
  gettimeofday(&start, NULL);
  std::cout << std::endl << "STARTING: " << std::endl;
  neuralplex::NeuralNet *neural_net = new neuralplex::NeuralNet(n_input, n_hidden, n_output, Sigmoid, SigmoidPrime);
  float global_error = neural_net->Train(*training_data_arr, batch_size, neuralplex::kLearningAlgorithmsResilientProp);
  if (global_error <= neuralplex::kNeuralLearningThreshold) {
    did_converge = true;
    gettimeofday(&end, NULL);
    elapsed_time +=   (end.tv_sec * (unsigned int)1e6 +   end.tv_usec) - (start.tv_sec * (unsigned int)1e6 + start.tv_usec);
  }
  if (did_converge) {
    std::cout << std::endl << "STATS: " << std::endl << (neural_net->epoch()) << " training intervals (epoch) to convergence." << std::endl;
    std::cout << (elapsed_time/1000) << "ms time to converge." << std::endl << std::endl;
    std::cout << "GENERATED NETWORK:" << std::endl;
    std::cout << neural_net->ToJSON() << std::endl << std::endl;
    std::cout << std::endl << "TEST RESULTS:" << std::endl;
    for(int i = 0; i < n_all_rows; i++) {
      float results[n_output];
      /*for(int j = 0; j < n_output; j++) {
        results[i] = 0.0f;
      }*/
      neural_net->Compute(&test_data_arr[i][0], &results[0]);
      std::cout << user_ids[i] << ",\"" <<  statuses[i] << "\",\"" <<  emails[i] << "\",";
      for(int j = 0; j < n_output; j++) {
        std::cout  << results[j] << ",";
      }
      std::cout  << std::endl;
    }
  }
  mysql_close(conn);
  return 0;
}

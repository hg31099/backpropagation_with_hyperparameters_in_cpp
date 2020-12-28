#include<bits/stdc++.h>
#include <sstream>
#include<fstream>
using namespace std;

double learning_rate=0.3;								// Specified learning rate and gamma values for input model
double Gamma=0.00001;

typedef struct NeuronStruct								// Defining the neuron structure
{
	vector<pair<int,double>> input;
	double output,delta,zin;
}Neuron;

/* Activation Functions and their derivatives' functions follow from here */

double sigmoid(double x)
{
     double exp_value,return_value;
     exp_value = exp((double) -x);
     return_value = 1 / (1 + exp_value);
     return return_value;
}

double fast_sigmoid(double x)
{
	double ret= x / (1 + abs(x));
	return  ret;
}

double relu(double x)
{
	double ret;
	if(x<=0) 
	{ret=0;}
	else {ret=x;}
	return ret;

}

double tanh(double x)
{
	double exp_value_minus,exp_value_plus,return_value;
    exp_value_minus = exp((double) -x);
	exp_value_plus = exp((double) x);
	return_value=(exp_value_plus-exp_value_minus)/(exp_value_plus+exp_value_minus);
	return return_value;
}

double identity(double x)
{
	return x;
}

double der_identity(double x)
{
	return 1;
}
double der_tanh(double x)
{
	double ret;
	ret=1-(x*x);
	return ret;
}
double der_relu(double x)
{
	double ret;
	if(x<=0)
	{
		ret=0;
	}
	else
	{
		ret=1;
	}
	return ret;
	
}

double der_sigmoid(double value)
{
	double ret;
	ret=(value)*(1-value);
	return ret;
}

double derivative(double value,int fxn_choice)
{
	double ret;
	switch (fxn_choice)
	{
	case 1:
		ret=der_sigmoid(value);
		break;
	case 2:
		ret=der_identity(value);
		break;
	case 3:
		ret=der_tanh(value);
		break;
	case 4:
		ret=der_relu(value);
		break;
	
	default: ret=1;
		break;
	}
	return ret;
}

/* Network Functions from here */
Neuron neuron_initialize()									// Initializing a new neuron in the network
{
	Neuron n;
	n.input.clear();
	n.output=1;
	n.zin=1;
	n.delta=0;
	return n;
}

double activate(vector<pair<int,double>> input,vector<Neuron> layer)		// Activating neurons (sum of all inputts and bias(weights multiplied)) 
{
	double value=0;
	for(auto kt=input.begin();kt!=input.end();kt++)
	{
		value+=((layer[kt->first].output)*(kt->second));
	}
	return value;
}

double activation(int fxn_choice, double value)								// Using activation functions per layer as speciified
{
	double ret;
	switch (fxn_choice)
	{
	case 1:
		ret=sigmoid(value);
		break;
	case 2:
		ret=identity(value);
		break;
	case 3:
		ret=tanh(value);
		break;
	case 4:
		ret=relu(value);
		break;
	
	default: ret=value;
		break;
	}
	return ret;
}

/* FeedForwarding the input values in the network*/
void feedforward(vector<vector<Neuron>> &network, vector<int> activation_function, int total_layers)
{
	int layer_count=1,l=0,layer_size,k=1;
	vector<Neuron> layer=network[0];
	for(auto it=network.begin()+1;it!=network.end();it++)
	{	
		layer_size=(*it).size();
		l=0;
		for(auto jt=(*it).begin();jt!=(*it).end();jt++)
		{
			if(l!=layer_size-1 && k!=network.size()-1)
			{
				(*jt).zin=activate((*jt).input,layer);	
				(*jt).output=activation(activation_function[layer_count-1],(*jt).zin);	
			}
			if(k==network.size()-1)
			{
				(*jt).zin=activate((*jt).input,layer);	
				(*jt).output=activation(activation_function[layer_count-1],(*jt).zin);	
			}
			l++;
		}
		layer_count++;
		k++;
		layer=*it;
	}
}

double error_cal(double x, double y)						// Absolute error calculated at the last layer
{
	return (x-y);
}

double error_mse(double x, double y)						// Mean Squared Error calculated after every epoch
{
	double ret=(x-y)*(x-y);
	return ret;
}

int p=0;
bool isEqual(const std::pair<int,double>& element)			// Checks if neuron (index p) is connected to an other neuron
{
    return element.first == p;
}

/* Backpropagates and updates weights after every input row*/
void backpropagate(vector<vector<Neuron>> &network, vector<double> expected,vector<int> activation_function)
{
	vector<double> errors;
	int layer_count=network.size();
	auto prev=network.rbegin();
	double error=0.0;
	int j=0;
	double deltaw=0.0;
	for(auto it=network.rbegin();it!=network.rend();it++)
	{
		errors.clear();
		if(layer_count!=network.size())
		{
			p=0;
			for(auto jt=(*it).begin();jt!=(*it).end();jt++)
			{
				error=0.0;
				for(auto kt=network[layer_count].begin();kt!=network[layer_count].end();kt++)
				{
					auto ft = find_if((*kt).input.begin(),(*kt).input.end(),isEqual);	
					if(ft!=(*kt).input.end())
					{
						error+=((*ft).second)*((*kt).delta);
						deltaw=(learning_rate)*((*kt).delta)*((*jt).output);
						(*ft).second=(*ft).second+deltaw;							
						(*ft).second=((*ft).second*(1-2*Gamma));

					}
				}
				errors.push_back(error);
				p++;	
			}
		}
		else
		{
			j=0;
			for(auto jt=(*it).begin();jt!=(*it).end();jt++)
			{
				errors.push_back(error_cal(expected[j],(*jt).output));
				j++;
			}

		}
		j=0;
		for(auto jt=(*it).begin();jt!=(*it).end();jt++)
		{
			(*jt).delta=errors[j]*derivative((*jt).output,activation_function[layer_count-1]);
			j++;
		}
		prev=it;
		layer_count--;
	}
}

/* Trains the network and finds error after every epoch */
void train_network(vector<vector<Neuron>> &network,vector<int> activation_function, int total_layers, int no_of_epochs,int no_of_neurons[],vector<double> expected_output,vector<vector<double>> data) 
{
	vector<double> expected_output_per_row;
	int i,j,k,l;
	double sum_error;
	
    for(i=0;i<no_of_epochs;i++)
    {
		sum_error=0.0;
        for(j=0;j<data.size();j++)
        {
            k=0;
            for(auto it = network[0].begin();it!=network[0].end()-1;it++)
            {
                (*it).zin=data[j][k];
				(*it).output=data[j][k];
                k++;
            }
            feedforward(network,activation_function,total_layers);
            expected_output_per_row.clear();
            for(k=0;k<no_of_neurons[total_layers-1];k++)
            {
                expected_output_per_row.push_back(0);
            }
            expected_output_per_row[expected_output[j]-1]=1;
			for(l=0;l<no_of_neurons[total_layers-1];l++)
			{
				sum_error+=error_mse(expected_output_per_row[l],network[total_layers-1][l].output);
			}
			backpropagate(network,expected_output_per_row,activation_function);
        }
		sum_error=((sum_error)/(double)no_of_neurons[total_layers-1]);

		cout<<"Epoch "<<i<<" completed. Error = "<<sum_error<<endl;
    }

	
}

/* Predicts output values for a test dataset*/
vector<int> predict(vector<vector<double>> data,vector<vector<Neuron>> &network,vector<int> activation_function, int total_layers,int no_of_neurons[])
{
	vector<int> predicted;
	int j,k,max_index,i;
	double max1;
	for(j=0;j<data.size();j++)
	{
		k=0;
		for(auto it = network[0].begin();it!=network[0].end()-1;it++)
		{
			(*it).zin=data[j][k];
			(*it).output=data[j][k];
			k++;
		}
		feedforward(network,activation_function,total_layers);
		max_index=0;
		max1=network[total_layers-1][0].output;
		// cout<<max1<<" ";
		for(i=1;i<network[total_layers-1].size();i++)
		{
			// cout<<network[total_layers-1][i].output<<" ";
			if(network[total_layers-1][i].output>max1)
			{
				max1=network[total_layers-1][i].output;
				max_index=i;
			}
		}
		// cout<<endl;
		predicted.push_back(max_index+1);
	}
	return(predicted);
}

vector<vector<double>> dataset_minmax(vector<vector<double>> data)		//Finds min and max values for every input in the dataset
{
	vector<vector<double>> minmax;
	minmax.clear();
	minmax.push_back(data[0]);
	minmax.push_back(data[0]);
	int i,j;
	for(i=1;i<data.size();i++)
	{
		for(j=0;j<data[0].size();j++)
		{
			if(data[i][j]<minmax[0][j])
			{
				minmax[0][j]=data[i][j];
			}
			if(data[i][j]>minmax[1][j])
			{
				minmax[1][j]=data[i][j];
			}
		}
	}
	return minmax;
}

void normalize_dataset(vector<vector<double>> &data,vector<vector<double>> minmax)		//Normalizes the dataset 
{
	int i,j;
	for(i=0;i<data.size();i++)
	{
		for(j=0;j<data[0].size();j++)
		{
			data[i][j]=((data[i][j]-minmax[0][j])/(double)(minmax[1][j]-minmax[0][j]));
		}
	}
}

double accuracy_metric(vector<int> predicted,vector<int> actual)			// Finds number of correctly predicted input cases
{
	int i=0,count=0;
	for(i=0;i<predicted.size();i++)
	{
		if(predicted[i]==actual[i])
		{
			count++;
		}
	}
	return((double)count/predicted.size());
}

/* Evaluates the algorithm using K Means Cross-Validation and Weight Decay regularization techniques */
void evaluate_algorithm(vector<vector<Neuron>> &network,vector<int> activation_function, int total_layers, int no_of_epochs,int no_of_neurons[],vector<double> &expected_output,vector<vector<double>> &data,int n_folds) 
{
	int rows_fold,i,endrow,j,k=0,l=0,m=0;
	double sum_scores;
	rows_fold=(data.size()/n_folds);
	vector<double> expected_output_copy;
	vector<vector<double>> data_copy;
	vector<int> predicted,actual;
	vector<double> scores;

	vector<vector<double>> data_predict;
	data_predict.clear();
	for(i=0;i<data.size();)
	{
		cout<<"Fold  "<<k+1<<"-----------------------------------------------------------------------"<<endl;
		expected_output_copy.clear();
		data_copy.clear();
		actual.clear();
		
		predicted.clear();

		if(k!=n_folds-1)
		{
			endrow=i+rows_fold-1;
		}
		else
		{
			endrow=data.size()-1;
		}
		cout<<"Testing  --- "<<endl;
		for(l=0;l<data.size();l++)
		{	
			if(!(l>=i && l<=endrow))
			{
				data_copy.push_back(data[l]);
				expected_output_copy.push_back(expected_output[l]);
			}
		}

		data_predict.clear();
		for(j=i;j<=endrow;j++)
		{
			data_predict.push_back(data[j]);
			actual.push_back(expected_output[j]);
		}

		train_network(network,activation_function,total_layers,no_of_epochs,no_of_neurons,expected_output_copy,data_copy);
		
		predicted =predict(data_predict,network,activation_function,total_layers,no_of_neurons);
		scores.push_back(accuracy_metric(predicted,actual));
		i=endrow+1;
		k++;
	}
	
	for(i=0;i<n_folds;i++)
	{
		sum_scores+=scores[i];
		cout<<"Epoch "<<(i+1)<<" Percent Accuracy "<<(scores[i]*100)<<endl;
	}
	cout<<"Mean Accuracy : "<<((sum_scores/n_folds)*100)<<endl;
}


int main()
{
	string line;
	char str[1000];
	ifstream fin,din,source;
	ofstream fout,dout;
	
	double weight;
	int no_of_hidden_layers,i,col,total_layers,j,choice,k,fxn_choice,node_index,no_of_epochs,sum_error,l,n_folds;
	vector<int> activation_function;
	
	cout<<"Enter no of hidden layers in the network : ";
	cin>>no_of_hidden_layers;	

	total_layers=no_of_hidden_layers+2;
	int no_of_neurons[total_layers];
	cout<<"Enter number of input values in 1st layer : ";
	cin>>no_of_neurons[0];
	cout<<"Enter number of neurons in each hidden layer : "<<endl;
	for(i=1;i<total_layers-1;i++)
	{
		cout<<"Layer "<<(i+1)<<" : ";
		cin>>no_of_neurons[i];
		cout<<endl;
	}
	no_of_neurons[total_layers-1]=3;

	din.open("Dataset.txt");				// Reads data from the input file, shuffles it and writes it back in the file
	vector<string> data_read;
	while(getline(din, line))
	{
		data_read.push_back(line);
	}
	din.close();
	random_shuffle(data_read.begin(),data_read.end());
	dout.open("Dataset.txt");
	
	for (i=0;i<data_read.size();i++)
	{
		dout<<data_read[i]<<endl;
	}
	dout.close();

	vector<double> v,expected_output;
	vector<vector<double>> data;
	set <double> s;	                   
	source.open("Dataset.txt", ios_base::in);  
	i=0;
	data.clear();
    expected_output.clear();								//Takes data input from the dataset file
	for(line;getline(source, line) ; )   
	{
		istringstream in(line);      
		double x,y;
		v.clear();
		for(i=0;i<no_of_neurons[0];i++)
		{
			in >> x;
			v.push_back(x);
		}
		in>>y;
		expected_output.push_back(y);
		s.insert(y);									
		data.push_back(v);
	}
													 
	no_of_neurons[total_layers-1]=s.size();
	
	cout<<"Enter 1 for fully connected network and 2 for reading user defined architecture from dataset file: ";
	cin>>choice;
	
	cout<<"Enter activation function for each layer (starting from 2nd layer): "<<endl<<"1 for Sigmoid Function"<<endl<<"2 for Identity Function"<<endl<<"3 for tanh function" <<endl<<"4 for Relu Function"<<endl;
	for(i=0;i<total_layers-1;i++)
	{
		cout<<"Choice for Layer "<<(i+1)<<" : ";
		cin>>fxn_choice;
		activation_function.push_back(fxn_choice);
	}
	
	if(choice == 1)										//Adding input for fully connected network in file 
	{
		fout.open("Network.txt");
		for(i=1;i<total_layers;i++)
		{
			for(k=0;k<no_of_neurons[i];k++)
			{
				for(j=0;j<no_of_neurons[i-1];j++)
				{
					fout<<j<<" ";
				}
				fout<<endl;
			}
		}
		fout.close();
	}

	fin.open("Network.txt");
	vector<vector<Neuron>> network(total_layers);
	vector<pair<int,double>> neuron_input;
	vector<Neuron> layer;
	network.clear();

	
	for(i=0;i<total_layers;i++)							//Initialized all neurons and creates the network 
	{
		layer.clear();
		for(j=0;j<no_of_neurons[i];j++)
		{
			if(i==0)
			{
				Neuron n=neuron_initialize();
				layer.push_back(n);
			}
			else
			{		
				getline(fin, line);
				strcpy(str,line.c_str());
				char *token = strtok(str, " ");
			    Neuron n=neuron_initialize();
				neuron_input.clear();
				while (token != NULL)
				{
					node_index = (int)(*token)-48;
					weight = ((double) rand() / (RAND_MAX));
					neuron_input.push_back({node_index,weight});
					token = strtok(NULL, " ");
				}
				weight = ((double) rand() / (RAND_MAX));
				neuron_input.push_back({no_of_neurons[i-1],weight});
				n.input=neuron_input;
				layer.push_back(n);
			}
		}
		if(i!=total_layers-1)
		{
			Neuron n=neuron_initialize();
			layer.push_back(n);
		}
		network.push_back(layer);
	}

//	-------------------------------------------------------------------------------------------------------
	cout<<"Enter number of epochs : ";
	cin>>no_of_epochs;
	vector<vector<double>> minmax=dataset_minmax(data);
	normalize_dataset(data,minmax);
	cout<<"Enter number of folds : ";
	cin>>n_folds;
	cout<<"Enter Learning Rate and Gamma (for weight decay) ";
	cin>>learning_rate>>Gamma;

	/* Evaluates using K-Cross Validation Dataset and Weight Decay*/
	evaluate_algorithm(network,activation_function,total_layers,no_of_epochs,no_of_neurons,expected_output,data,n_folds);
	
	return 0;
}
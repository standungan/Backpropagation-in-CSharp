using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace BackpropagationNet
{
    public partial class Form1 : Form
    {
        //--------------------------------------------------
        public Double[,] data_training = new Double[63, 21];
        public Double[,] data_target = new Double[7, 21];
        public String[] data_huruf = new String[21];
        public Double[] error = new Double[21];

        public static Double[] x = new Double[64];
        public static Double[] y = new Double[7];
        public static Double[] z = new Double[7];

        public Double[,] v = new Double[x.GetLength(0), z.GetLength(0)];
        public Double[,] delta_V = new Double[x.GetLength(0), z.GetLength(0)];
        public Double[,] v_new = new Double[x.GetLength(0), z.GetLength(0)];
        public Double[,] v_old = new Double[x.GetLength(0), z.GetLength(0)];
        
        public Double[,] w = new Double[z.GetLength(0), y.GetLength(0)];
        public Double[,] delta_W = new Double[z.GetLength(0), y.GetLength(0)];
        public Double[,] w_new = new Double[z.GetLength(0), y.GetLength(0)];
        public Double[,] w_old = new Double[z.GetLength(0), y.GetLength(0)];

        public Double[] x_v  = new Double[z.GetLength(0)];
        public Double[] z_in = new Double[z.GetLength(0)];
        public Double[] delta_in = new Double[z.GetLength(0)];
        public Double[] delta_hidden = new Double[z.GetLength(0)];
 
        public Double[] z_w = new Double[y.GetLength(0)];
        public Double[] delta_output = new Double[y.GetLength(0)];
        public Double[] y_in = new Double[y.GetLength(0)];
        public Double[] t = new Double[y.GetLength(0)];

        public Double[] x_uji = new Double[x.GetLength(0)];

        
        Double SE = 0;
        Double max_error = 0;
        Random rand = new Random();
        Double max = 0.1;
        Double min = -0.1;
        Boolean kondisi;
        int pairs;
        Double alpha = 0;
        int epoch;
        Double target_error;
        int max_iteration = 300;

        //---------------------------------------------------------------------
        public Form1()
        {
            InitializeComponent();
        }

        public void backpropagation() // Referensi : Fundamental Of Neural Networks by Laurene Fausett
        {
            // Step 0. initialize weight, set to small random values
            
            for(int j = 0; j < v.GetLength(1); j++)
            {
                for (int i = 0; i < v.GetLength(0); i++)
                {
                    v[i, j] = rand.NextDouble() * (max - min) + min;
	            }
            }

            for(int k = 0; k < w.GetLength(1); k++ )
            {
	            for(int j = 0; j < w.GetLength(0); j++)
                {
                    w[j, k] = rand.NextDouble() * (max - min) + min;
	            }
            }
            epoch = 0;
            kondisi = false;
            // step 1 while stopping condition is false, do Steps 2-9. 
            while(kondisi == false)
            {
                epoch = epoch + 1;
                //step 2 For each training pair, do Steps 3-8.
	            for(int data = 0; data < pairs; data++)
                {
		            //----------------------------------------------------------------
		            // Feed Forward
		            //----------------------------------------------------------------
		            //Step 3 Each input unit (Xi, i = 1, ... , n) 
		            //receives input signal Xi and broadcasts 
		            //this signal to all units in the layer above (the hidden units).
		            
                    for(int i = 1; i < x.GetLength(0); i++)
		            {
			            x[i] =  data_training[i-1,data];
		            }
		        
                		
		            for(int k = 0; k < t.GetLength(0); k++)
		            {
			            t[k] = data_target[k,data];	
		            }
		            //----------------------------------------------------------------
		            //Step 4, Each hidden unit (Zj,j = 1, ... ,36) sums its weighted input signals,
                    
		            for(int j = 1; j < v.GetLength(1); j++)
                    {
			            //menghitung sigma(xi*vij)
                        x_v[j] = 0;
                        for(int i = 1; i < x.GetLength(0); i++)
                        {
                            x_v[j] = x_v[j] + (x[i] * v[i, j]);
			            }
		            }

                    for (int j = 1; j < z_in.GetLength(0); j++)
                    {
                        z_in[j] = v[0, j] + x_v[j];
                    }

                    z[0] = 1;
                    for (int j = 1; j < z.GetLength(0); j++)
                    {
                        z[j] = fungsiaktivasi(z_in[j]);
                    }
		            //--------------------------------------------------------------
		            //step 5 Each output unit (Yk , k = 1, ... , m) sums its weighted input signals,
		            for(int k = 0 ; k < z_w.GetLength(0); k++)
                    {
                        z_w[k] = 0;
			            for(int j = 1; j < z.GetLength(0); j++)
                        {
				            z_w[k] = z_w[k] + (z[j] * w[j,k]);
			            }
                        
                     }

                    for (int k = 0; k < y_in.GetLength(0); k++)
                    {
                        y_in[k] = w[0, k] + z_w[k];
                    }

                    for (int k = 0; k < y.GetLength(0); k++)
                    {
                        y[k] = fungsiaktivasi(y_in[k]);
                    }
                	//----------------------------------------------------------------
		            
                    // stopping condition 
		            // sigma ((t-y)^2) / 2 <= target error
                    SE = 0;
                    for (int k = 0; k < y.GetLength(0); k++ )
                    {
                        SE = SE + ((t[k] - y[k]) * (t[k] - y[k]));
                        //SE = SE + ((t[k] - y[k]) * (t[k] - y[k]));
                    }
                    error[data] = SE / 2;
		            //----------------------------------------------------------------
		            // Backpropagation
		            //----------------------------------------------------------------
                		
		            //step 6 Each output unit (Yk , k = 1, ... ,m)
		            //receives a target pattern corresponding to the input training pattern, 
		            //computes its error information term,
		            for(int k =0; k < delta_output.GetLength(0); k++)
                    {
			            delta_output[k] = (t[k] - y[k])*turunanaktivasi(y_in[k]);
		            }
		            //calculates its weightcorrection term (used to update Wjk later),
		            for(int k = 0; k < delta_output.GetLength(0); k++)
                    {
    			        for(int j = 1; j < z.GetLength(0); j++)
                        {
	    			        delta_W[j,k] = alpha * delta_output[k] * z[j];
		    	        }
		            }
                		
		            for(int k =0; k < delta_output.GetLength(0); k++)
                    {
			            delta_W[0,k] = alpha * delta_output[k];			
		            }
		            //----------------------------------------------------------------
                		
		            // step 7 Each hidden unit (Zj, j = 1, ... ,p) 
		            // sums its delta inputs (from units in the layer above),
		            for(int j = 1; j < delta_in.GetLength(0); j++)
                    {
                        delta_in[j] = 0;
                        for (int k = 0; k < delta_output.GetLength(0); k++)
                        {
	    			        delta_in[j] = delta_in[j] + (delta_output[k] * w[j,k]);
		    	        }
		            }
		
		            //multiplies by the derivative of its activation function 
		            //to calculate its error information term,
		            for(int j = 1; j < delta_hidden.GetLength(0); j++)
                    {
			            delta_hidden[j] = delta_in[j] * turunanaktivasi(z_in[j]);			
		            }
                		
		            //calculates its weight correction term (used to update vij later),
		            for(int i = 1; i < x.GetLength(0); i++){
			            for(int j = 1; j < delta_hidden.GetLength(0); j++){
				            delta_V[i,j] = alpha * delta_hidden[j] * x[i];
			            }
		            }
		
		            //and calculates its bias correction term (used to update VOj later),
		            for(int j = 1; j < delta_hidden.GetLength(0) ; j++){
			            delta_V[0,j] = alpha * delta_hidden[j];
		            }
                		
		            //----------------------------------------------------------------
		            // update weight and bias
		            //----------------------------------------------------------------
		            // step 8 
		            w_old = w;
		            v_old = v;
		            //Each output unit (Yk, k = I, , m) updates its bias and weights (j = 0, , p): 
		            for(int k = 0; k < w_new.GetLength(1) ; k++){
    			        for(int j = 0; j < w_new.GetLength(0); j++){
	    			        w_new[j,k] = w_old[j,k] + delta_W[j,k];
		    	        }
		            }
		            //Each hidden unit (Z], j == 1, ,p) updates its bias and weights (i = 0, , n): 
		            for(int j = 0; j < v_new.GetLength(1); j++){
                        for (int i = 0; i < v_new.GetLength(0); i++)
                        {
                            v_new[i, j] = v_old[i, j] + delta_V[i, j];
                        }
		            }
                		
		            w = w_new;
		            v = v_new;
		            //----------------------------------------------------------------------------
                		
		            //End of Step 2
	            }
	            //Step 9 Test Stopping condition
		        // 1. selisih w_new - w_old 
		        // 2. sigma ((t-y)^2) / 2 <= target error
		        // 3. maximum iterasi
                Console.WriteLine(epoch);
                //richTextBox1.AppendText("Epoch : " + Convert.ToString(epoch));
                max_error = error[0];
                for (int i = 0; i < pairs; i++)
                {
                    if (error[i] > max_error)
                    {
                        max_error = error[i];
                    }
                }
                if (max_error <= target_error){
                    kondisi = true;
                }
                
                if(epoch == max_iteration){
                    kondisi = true;
                }
                //richTextBox1.AppendText("Error : " + Convert.ToString(max_error));
                Console.WriteLine(max_error);
                progressBar1.Increment(1); 
            }
        }

        public double fungsiaktivasi( double a)
        { 
            //return 1 / (1 + Math.Exp(-a)); ;
            return 2 / (1 + Math.Exp(-2 * a)) - 1;
        }

        public double turunanaktivasi(double a)
        {
            double fx = fungsiaktivasi(a);

            //return fx * (1 - fx);
            return (1 - Math.Pow(fx, 2)) / 2;
        }

        public void reset_checkbox()
        {
            foreach (CheckBox c in groupBox1.Controls)
            {

                c.Checked = false;
                c.BackColor = System.Drawing.SystemColors.Control;

            }

            foreach (CheckBox c2 in groupBox2.Controls)
            {
                c2.Checked = false;
                c2.BackColor = System.Drawing.SystemColors.Control;
            }
        }

        public static int detectnumber(string a)
        {
            string b = string.Empty;
            int val = 0;

            for (int i = 0; i < a.Length; i++)
            {
                if (Char.IsDigit(a[i]))
                    b += a[i];
            }

            if (b.Length > 0)
                val = int.Parse(b);

            return val;
        }

        private void inputToolStripMenuItem_Click(object sender, EventArgs e)
        {
            inputpanel.Visible = true;
            trainpanel.Visible = false;
            testpanel.Visible = false;
        }

        private void trainingToolStripMenuItem_Click(object sender, EventArgs e)
        {
            inputpanel.Visible = false;
            trainpanel.Visible = true;
            testpanel.Visible = false;
        }

        private void testingToolStripMenuItem_Click(object sender, EventArgs e)
        {
            inputpanel.Visible = false;
            trainpanel.Visible = false;
            testpanel.Visible = true;
        }

        private void Target1_CheckedChanged(object sender, EventArgs e)
        {
            if (Target1.Checked)
            {
                Target1.BackColor = System.Drawing.SystemColors.ControlDarkDark;
                data_huruf[pairs] = "A";
                Target2.Enabled = false; Target3.Enabled = false; Target4.Enabled = false;
                Target5.Enabled = false; Target6.Enabled = false; Target7.Enabled = false;
            }
            else
            {
                Target1.BackColor = System.Drawing.SystemColors.Control;
                Target2.Enabled = true; Target3.Enabled = true; Target4.Enabled = true;
                Target5.Enabled = true; Target6.Enabled = true; Target7.Enabled = true;
            }
        }

        private void Target2_CheckedChanged(object sender, EventArgs e)
        {
            if (Target2.Checked)
            {
                Target2.BackColor = System.Drawing.SystemColors.ControlDarkDark;
                data_huruf[pairs] = "B";
                Target1.Enabled = false; Target3.Enabled = false; Target4.Enabled = false;
                Target5.Enabled = false; Target6.Enabled = false; Target7.Enabled = false;
            }
            else
            {
                Target2.BackColor = System.Drawing.SystemColors.Control;
                Target1.Enabled = true; Target3.Enabled = true; Target4.Enabled = true;
                Target5.Enabled = true; Target6.Enabled = true; Target7.Enabled = true;
            }
        }

        private void Target3_CheckedChanged(object sender, EventArgs e)
        {
            if (Target3.Checked)
            {
                Target3.BackColor = System.Drawing.SystemColors.ControlDarkDark;
                data_huruf[pairs] = "C";
                Target1.Enabled = false; Target2.Enabled = false; Target4.Enabled = false;
                Target5.Enabled = false; Target6.Enabled = false; Target7.Enabled = false;
            }
            else
            {
                Target3.BackColor = System.Drawing.SystemColors.Control;
                Target1.Enabled = true; Target2.Enabled = true; Target4.Enabled = true;
                Target5.Enabled = true; Target6.Enabled = true; Target7.Enabled = true;
            }
        }

        private void Target4_CheckedChanged(object sender, EventArgs e)
        {
            if (Target4.Checked)
            {
                Target4.BackColor = System.Drawing.SystemColors.ControlDarkDark;
                data_huruf[pairs] = "D";
                Target1.Enabled = false; Target2.Enabled = false; Target3.Enabled = false;
                Target5.Enabled = false; Target6.Enabled = false; Target7.Enabled = false;
            }
            else
            {
                Target4.BackColor = System.Drawing.SystemColors.Control;
                Target1.Enabled = true; Target2.Enabled = true; Target3.Enabled = true;
                Target5.Enabled = true; Target6.Enabled = true; Target7.Enabled = true;
            }
        }

        private void Target5_CheckedChanged(object sender, EventArgs e)
        {
            if (Target5.Checked)
            {
                Target5.BackColor = System.Drawing.SystemColors.ControlDarkDark;
                data_huruf[pairs] = "E";
                Target1.Enabled = false; Target2.Enabled = false; Target3.Enabled = false;
                Target4.Enabled = false; Target6.Enabled = false; Target7.Enabled = false;
            }
            else
            {
                Target5.BackColor = System.Drawing.SystemColors.Control;
                Target1.Enabled = true; Target2.Enabled = true; Target3.Enabled = true;
                Target4.Enabled = true; Target6.Enabled = true; Target7.Enabled = true;
            }
        }

        private void Target6_CheckedChanged(object sender, EventArgs e)
        {
            if (Target6.Checked)
            {
                Target6.BackColor = System.Drawing.SystemColors.ControlDarkDark;
                data_huruf[pairs] = "J";
                Target1.Enabled = false; Target2.Enabled = false; Target3.Enabled = false;
                Target4.Enabled = false; Target5.Enabled = false; Target7.Enabled = false;
            }
            else
            {
                Target6.BackColor = System.Drawing.SystemColors.Control;
                Target1.Enabled = true; Target2.Enabled = true; Target3.Enabled = true;
                Target4.Enabled = true; Target5.Enabled = true; Target7.Enabled = true;
            }
        }

        private void Target7_CheckedChanged(object sender, EventArgs e)
        {
            if (Target7.Checked)
            {
                Target7.BackColor = System.Drawing.SystemColors.ControlDarkDark;
                data_huruf[pairs] = "K";
                Target1.Enabled = false; Target2.Enabled = false; Target3.Enabled = false;
                Target4.Enabled = false; Target5.Enabled = false; Target6.Enabled = false;
            }
            else
            {
                Target7.BackColor = System.Drawing.SystemColors.Control;
                Target1.Enabled = true; Target2.Enabled = true; Target3.Enabled = true;
                Target4.Enabled = true; Target5.Enabled = true; Target6.Enabled = true;
            }
        }

        private void checkBox1_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox1.Checked)
                checkBox1.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox1.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox2_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox2.Checked)
                checkBox2.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox2.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox3_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox3.Checked)
                checkBox3.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox3.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox4_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox4.Checked)
                checkBox4.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox4.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox5_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox5.Checked)
                checkBox5.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox5.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox6_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox6.Checked)
                checkBox6.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox6.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox7_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox7.Checked)
                checkBox7.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox7.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox8_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox8.Checked)
                checkBox8.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox8.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox9_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox9.Checked)
                checkBox9.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox9.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox10_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox10.Checked)
                checkBox10.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox10.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox11_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox11.Checked)
                checkBox11.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox11.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox12_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox12.Checked)
                checkBox12.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox12.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox13_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox13.Checked)
                checkBox13.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox13.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox14_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox14.Checked)
                checkBox14.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox14.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox15_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox15.Checked)
                checkBox15.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox15.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox16_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox16.Checked)
                checkBox16.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox16.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox17_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox17.Checked)
                checkBox17.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox17.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox18_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox18.Checked)
                checkBox18.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox18.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox19_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox19.Checked)
                checkBox19.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox19.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox20_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox20.Checked)
                checkBox20.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox20.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox21_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox21.Checked)
                checkBox21.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox21.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox22_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox22.Checked)
                checkBox22.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox22.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox23_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox23.Checked)
                checkBox23.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox23.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox24_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox24.Checked)
                checkBox24.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox24.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox25_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox25.Checked)
                checkBox25.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox25.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox26_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox26.Checked)
                checkBox26.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox26.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox27_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox27.Checked)
                checkBox27.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox27.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox28_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox28.Checked)
                checkBox28.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox28.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox29_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox29.Checked)
                checkBox29.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox29.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox30_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox30.Checked)
                checkBox30.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox30.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox31_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox31.Checked)
                checkBox31.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox31.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox32_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox32.Checked)
                checkBox32.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox32.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox33_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox33.Checked)
                checkBox33.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox33.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox34_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox34.Checked)
                checkBox34.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox34.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox35_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox35.Checked)
                checkBox35.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox35.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox36_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox36.Checked)
                checkBox36.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox36.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox37_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox37.Checked)
                checkBox37.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox37.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox38_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox38.Checked)
                checkBox38.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox38.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox39_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox39.Checked)
                checkBox39.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox39.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox40_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox40.Checked)
                checkBox40.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox40.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox41_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox41.Checked)
                checkBox41.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox41.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox42_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox42.Checked)
                checkBox42.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox42.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox43_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox43.Checked)
                checkBox43.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox43.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox44_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox44.Checked)
                checkBox44.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox44.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox45_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox45.Checked)
                checkBox45.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox45.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox46_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox46.Checked)
                checkBox46.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox46.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox47_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox47.Checked)
                checkBox47.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox47.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox48_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox48.Checked)
                checkBox48.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox48.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox49_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox49.Checked)
                checkBox49.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox49.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox50_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox50.Checked)
                checkBox50.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox50.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox51_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox51.Checked)
                checkBox51.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox51.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox52_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox52.Checked)
                checkBox52.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox52.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox53_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox53.Checked)
                checkBox53.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox53.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox54_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox54.Checked)
                checkBox54.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox54.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox55_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox55.Checked)
                checkBox55.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox55.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox56_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox56.Checked)
                checkBox56.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox56.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox57_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox57.Checked)
                checkBox57.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox57.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox58_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox58.Checked)
                checkBox58.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox58.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox59_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox59.Checked)
                checkBox59.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox59.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox60_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox60.Checked)
                checkBox60.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox60.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox61_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox61.Checked)
                checkBox61.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox61.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox62_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox62.Checked)
                checkBox62.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox62.BackColor = System.Drawing.SystemColors.Control;
        }

        private void checkBox63_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBox63.Checked)
                checkBox63.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            else
                checkBox63.BackColor = System.Drawing.SystemColors.Control;
        }

        private void clearBTN_Click(object sender, EventArgs e)
        {
            reset_checkbox(); //memanggil method reset_checkbox()
        }

        private void clear2BTN_Click(object sender, EventArgs e)
        {
            reset_checkbox();
        }

        private void inputBTN_Click(object sender, EventArgs e)
        {
            foreach (Control con in groupBox1.Controls)
            {
                String a = string.Empty;
                int i = 0;
                a = con.Name;
                i = detectnumber(a);

                if (con is CheckBox && ((CheckBox)con).Checked)
                {
                    data_training[i - 1, pairs] = 1;
                }
                else if (con is CheckBox && !((CheckBox)con).Checked)
                {
                    data_training[i - 1, pairs] = -1;
                }
            }

            foreach (Control con in groupBox2.Controls)
            {
                String a = string.Empty;
                int i = 0;
                a = con.Name;
                i = detectnumber(a);

                if (con is CheckBox && ((CheckBox)con).Checked)
                {
                    data_target[i - 1, pairs] = 1;
                }
                else if (con is CheckBox && !((CheckBox)con).Checked)
                {
                    data_target[i - 1, pairs] = -1;
                }
            }

            pairs = pairs + 1;
            MessageBox.Show("Data Terinput");
        }

        private void trainBTN_Click(object sender, EventArgs e)
        {
            max_iteration = int.Parse(textBox3.Text); //maximum iteration/epoch
            alpha = Convert.ToDouble(textBox1.Text); //learning rate
            target_error = Convert.ToDouble(textBox2.Text); //target error
            progressBar1.Maximum = max_iteration; 
			
            backpropagation(); // memanggil method backpropagation()
            
			MessageBox.Show("Training Done");
            progressBar1.Value = 0;
        }

        private void testBTN_Click(object sender, EventArgs e)
        {            
            foreach (Control con in groupBox1.Controls)
            {
                String a = string.Empty;
                int i = 0;
                a = con.Name;
                i = detectnumber(a);

                if (con is CheckBox && ((CheckBox)con).Checked)
                {
                    x_uji[i] = 1;
                }
                else if (con is CheckBox && !((CheckBox)con).Checked)
                {
                    x_uji[i] = -1;
                }
            }


            //--------------------------------------------
            //Step 4, Each hidden unit (Zj,j = 1, ... ,36) 
			//sums its weighted input signals,
            z[0] = 1;
            for (int j = 1; j < v.GetLength(1); j++)
            {
                //menghitung sigma(xi*vij)
                x_v[j] = 0;
                for (int i = 1; i < v.GetLength(0); i++)
                {
                    x_v[j] = x_v[j] + (x_uji[i] * v[i, j]);
                }
            }

            for (int j = 1; j < z_in.GetLength(0); j++)
            {
                z_in[j] = v[0, j] + x_v[j];
            }

            for (int j = 1; j < z.GetLength(0); j++)
            {
                z[j] = fungsiaktivasi(z_in[j]);
            }
            //--------------------------------------------------------------
            //step 5 Each output unit (Yk , k = 1, ... , m) sums its weighted input signals,
            for (int k = 0; k < w.GetLength(1); k++)
            {
                z_w[k] = 0;
                for (int j = 1; j < w.GetLength(0); j++)
                {
                    z_w[k] = z_w[k] + (z[j] * w[j, k]);
                }
            }

            for (int k = 0; k < y_in.GetLength(0); k++)
            {
                y_in[k] = w[0, k] + z_w[k];

            }

            for (int k = 0; k < y.GetLength(0); k++)
            {
                y[k] = fungsiaktivasi(y_in[k]);
            }
            //--------------------------------------------------------------
            Double max_val = y[0];
            int pos = 0;
            for (int k = 0; k < y.GetLength(0); k++)
            {
                if (y[k] > max_val)
                {
                    max_val = y[k];
                    pos = k;
                }
            }

            if (pos == 0)
            {
                MessageBox.Show("Huruf : A");
            }
            else if (pos == 1)
            {
                MessageBox.Show("Huruf : B");
            }
            else if (pos == 2)
            {
                MessageBox.Show("Huruf : C");
            }
            else if (pos == 3)
            {
                MessageBox.Show("Huruf : D");
            }
            else if (pos == 4)
            {
                MessageBox.Show("Huruf : E");
            }
            else if (pos == 5)
            {
                MessageBox.Show("Huruf : J");
            }
            else if (pos == 6)
            {
                MessageBox.Show("Huruf : K");
            }
            
        }
    
    }
}
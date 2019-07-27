package com.sankusu2525.stepalarm;

import android.content.Context;
import android.content.SharedPreferences;
import android.util.Log;

/**
 * Created by .cipretec 2017/02/15
 */
public class SettingIO {
    private static final String TAG = "SettingIO";
    private boolean LogOutput =false;

    private SharedPreferences pref;
    private int mArrayLength;
    public float [] mArrayValue;
    public int CreateCount;

    public SettingIO(Context context , int arrayLength){
        String FunctionNameStr = "SettingIO";
        if(LogOutput) Log.d(TAG, FunctionNameStr + " S:");
        pref = context.getSharedPreferences("Setting", Context.MODE_PRIVATE);
        mArrayValue =new float[arrayLength];
        Read();

        if(LogOutput) Log.d(TAG, FunctionNameStr + " E:");
    }

    public void Read(){
        String FunctionNameStr = "Read";
        if(LogOutput) Log.d(TAG, FunctionNameStr + " S:");
        CreateCount =pref.getInt("Count",0);
        for (int i = 0; i <mArrayValue.length ; i++) {
            mArrayValue[i]=pref.getFloat("" +i +":",0);
        }

        if(LogOutput) Log.d(TAG, FunctionNameStr + " E:");
    }

    public void Write(){
        String FunctionNameStr = "Write";
        if(LogOutput) Log.d(TAG, FunctionNameStr + " S:");
        SharedPreferences.Editor editor = pref.edit();
        CreateCount ++;
        editor.putInt("Count",CreateCount);
        for (int i = 0; i <mArrayValue.length ; i++) {
            editor.putFloat("" +i +":",mArrayValue[i]);
        }
        editor.commit();
        if(LogOutput) Log.d(TAG, FunctionNameStr + " E:");
    }

    public void ArraySetting(float[] ArrayList){
        String FunctionNameStr = "ArraySetting";
        if(LogOutput) Log.d(TAG, FunctionNameStr + " S:");
        mArrayValue =ArrayList.clone();
        if(LogOutput) Log.d(TAG, FunctionNameStr + " E:");
    }

    public void ArraySetting(double[] ArrayList){
        String FunctionNameStr = "ArraySetting";
        if(LogOutput) Log.d(TAG, FunctionNameStr + " S:");
        for (int i = 0; i <mArrayValue.length ; i++) {
            mArrayValue[i] =(float)ArrayList[i];
        }
        if(LogOutput) Log.d(TAG, FunctionNameStr + " E:");
    }

    public void AppSetting(){
        String FunctionNameStr = "AppSetting";
        if(LogOutput) Log.d(TAG, FunctionNameStr + " S:");
        if(LogOutput) Log.d(TAG, FunctionNameStr + " E:");
    }

}
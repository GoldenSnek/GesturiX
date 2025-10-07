import React, { useEffect } from 'react';
import { View, Text, ActivityIndicator } from 'react-native';
import { supabase } from '../../src/supabaseClient';
import { router, useLocalSearchParams } from 'expo-router';

export default function OAuthCallback() {
  const params = useLocalSearchParams();

  // CRITICAL DEBUGGING LOG: Log the received parameters immediately.
  // This helps confirm if the app is receiving tokens or an empty object.
  console.log('--- OAUTH CALLBACK EXECUTING ---');
  console.log('Received Params:', JSON.stringify(params));

  useEffect(() => {
    const handleOAuth = async () => {
      try {
        const access_token = params['access_token'] as string;
        const refresh_token = params['refresh_token'] as string | undefined;

        if (!access_token) {
          
          // ADDED LOG: Detailed failure message to diagnose the redirect problem.
          console.error('🚨 OAUTH FAILURE: No access_token found in redirect parameters.');
          console.error('Full Redirect Params on Failure:', JSON.stringify(params));
          
          if (params.error) {
            console.error('Supabase Error:', params.error);
          }
          
          // Redirect to login on failure
          router.replace('/(stack)/Login');
          return;
        }

        // --- Session handling begins here (only if access_token is present) ---

        const { data: sessionData, error } = await supabase.auth.setSession({
          access_token,
          refresh_token: refresh_token ?? '',
        });

        if (error) {
          console.error('🚨 Supabase setSession error:', error);
          router.replace('/(stack)/Login');
          return;
        }

        const session = sessionData?.session;
        if (!session?.user) {
          console.error('🚨 setSession successful, but no user found in session data.');
          router.replace('/(stack)/Login');
          return;
        }

        const user = session.user;
        
        // 1. CHECK IF PROFILE EXISTS
        const { data: existingProfile, error: checkError } = await supabase
          .from('profiles')
          .select('id')
          .eq('id', user.id)
          .maybeSingle();

        if (checkError) {
             console.error('Error checking profile:', checkError);
             router.replace('/(stack)/Login');
             return;
        }
        
        // 2. LOGIC BRANCHING
        if (!existingProfile) {
             // New user via OAuth
             console.log('User logged in via OAuth but profile is missing. Redirecting to Sign Up.');
             router.replace('/(stack)/SignUp'); 
             return;

        } else {
             // Returning user
             console.log('✅ Returning OAuth user logged in. Redirecting...');
             router.replace('/(tabs)/translate');
             return;
        }

      } catch (err) {
        console.error('🚨 Unexpected OAuth error in handleOAuth:', err);
        router.replace('/(stack)/Login');
      }
    };

    if (params['access_token'] || Object.keys(params).length > 0) {
      // Run the handler if an access_token is detected OR if any params are received
      handleOAuth();
    } else {
        // Fallback for cases where the app is loaded directly without params
        console.log('ℹ️ OAuthCallback loaded without parameters. Redirecting to Login.');
        router.replace('/(stack)/Login');
    }
  }, [params]);

  return (
    <View className="flex-1 justify-center items-center bg-white">
      <ActivityIndicator size="large" color="#0D47A1" />
      <Text className="text-lg font-bold text-gray-700 mt-4">
        Finishing Google Sign In...
      </Text>
    </View>
  );
}
import React, { useEffect, useState } from 'react';
import { View, Text, TouchableOpacity, Image, ActivityIndicator } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { supabase } from '../src/supabaseClient';

interface ProfileData {
  id: string;
  username: string;
  email: string;
  created_at: string;
  photo_url?: string | null;
}

const AppHeaderProfile = () => {
  const insets = useSafeAreaInsets();
  const router = useRouter();

  const [profile, setProfile] = useState<ProfileData | null>(null);
  const [photoUrl, setPhotoUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchProfile = async () => {
      try {
        const {
          data: { user },
          error: userError,
        } = await supabase.auth.getUser();

        if (userError || !user) throw userError || new Error('No user logged in');

        const { data, error } = await supabase
          .from('profiles')
          .select('id, username, email, created_at, photo_url')
          .eq('id', user.id)
          .single();

        if (error) throw error;

        setProfile(data as ProfileData);

        if (data.photo_url) {
          const { data: urlData } = supabase.storage
            .from('avatars')
            .getPublicUrl(data.photo_url);

          if (urlData?.publicUrl) {
            setPhotoUrl(urlData.publicUrl);
          }
        }
      } catch (err: any) {
        console.error('Error fetching profile header:', err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchProfile();
  }, []);

  return (
    <View>
      <LinearGradient
        colors={['#FF6B00', '#FFAB7B']}
        className="py-3 px-4 flex-row items-center justify-center"
      >

        {loading ? (
          <ActivityIndicator size="small" color="#fff" />
        ) : profile ? (
          <View className="flex-row items-center">
            <Image
              source={
                photoUrl
                  ? { uri: photoUrl }
                  : require('../assets/images/CatPFP.jpg')
              }
              className="w-14 h-14 rounded-full mr-3"
            />
            <View>
              <Text className="text-primary text-lg font-fredoka-semibold">
                {profile.username}
              </Text>
              <Text className="text-primary text-xs font-fredoka">{profile.email}</Text>
              <Text className="text-primary text-xs font-fredoka">
                Member since {new Date(profile.created_at).toLocaleDateString()}
              </Text>
            </View>
          </View>
        ) : (
          <Text className="text-white text-sm">No profile data</Text>
        )}

        <View style={{ width: 26 }} />
      </LinearGradient>

      <LinearGradient
        colors={[
          'rgba(255, 171, 123, 1.0)',
          'rgba(255, 171, 123, 0.0)',
        ]}
        start={{ x: 0.5, y: 0.0 }}
        end={{ x: 0.5, y: 1.0 }}
        className="h-3"
      />
    </View>
  );
};

export default AppHeaderProfile;
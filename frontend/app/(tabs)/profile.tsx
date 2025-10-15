import React, { useEffect, useState, useRef } from 'react';
import {
  View,
  Text,
  Switch,
  TouchableOpacity,
  ScrollView,
  Image,
  ActivityIndicator,
  PanResponder,
  TextInput,
  Alert,
  Pressable,
  Modal,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { MaterialIcons, AntDesign } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { supabase } from '../../src/supabaseClient';
import AppHeaderProfile from '../../components/AppHeaderProfile';
import * as ImagePicker from 'expo-image-picker';
import * as FileSystem from 'expo-file-system';
import { Buffer } from 'buffer';
global.Buffer = global.Buffer || Buffer;

interface ProfileData {
  id: string;
  username: string;
  email: string;
  created_at: string;
  photo_url?: string | null;
}

const Profile = () => {
  const insets = useSafeAreaInsets();
  const [profile, setProfile] = useState<ProfileData | null>(null);
  const [photoUrl, setPhotoUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const [showPhotoModal, setShowPhotoModal] = useState(false);
  const [showPhotoOptions, setShowPhotoOptions] = useState(false);

  const [isEditingUsername, setIsEditingUsername] = useState(false);
  const [newUsername, setNewUsername] = useState('');
  const [usernameLoading, setUsernameLoading] = useState(false);

  const [isSoundEffectsEnabled, setSoundEffectsEnabled] = useState(false);
  const [isVibrationEnabled, setVibrationEnabled] = useState(false);
  const [isDarkModeEnabled, setDarkModeEnabled] = useState(false);
  const [areNotificationsEnabled, setNotificationsEnabled] = useState(false);

  const progressData = {
    lessonsCompleted: 25,
    signsLearned: 180,
    daysStreak: 7,
    practiceHours: 12.5,
  };

  const router = useRouter();

  const panResponder = useRef(
    PanResponder.create({
      onMoveShouldSetPanResponder: (_, gestureState) =>
        Math.abs(gestureState.dx) > 10,
      onPanResponderRelease: (_, gestureState) => {
        if (gestureState.dx > 30) {
          router.push('/learn');
        }
      },
    })
  ).current;

  useEffect(() => {
    const fetchProfile = async () => {
      try {
        setLoading(true);
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
        setNewUsername(data.username);

        if (data.photo_url) {
          const { data: urlData } = supabase.storage
            .from('avatars')
            .getPublicUrl(data.photo_url);
          setPhotoUrl(urlData?.publicUrl || null);
        } else {
          setPhotoUrl(null);
        }
      } catch (err: any) {
        console.error('Error fetching profile:', err.message);
      } finally {
        setLoading(false);
      }
    };
    fetchProfile();
  }, []);

  const handleSignOut = async () => {
    await supabase.auth.signOut();
    router.replace('/(stack)/LandingPage');
  };

  const handleSaveUsername = async () => {
    if (newUsername.trim().length < 3) {
      Alert.alert('Username must be at least 3 characters.');
      return;
    }
    setUsernameLoading(true);
    const { error } = await supabase
      .from('profiles')
      .update({ username: newUsername.trim(), updated_at: new Date().toISOString() })
      .eq('id', profile?.id);
    if (error) {
      Alert.alert('Update failed', error.message);
    } else {
      setProfile((prev) =>
        prev ? { ...prev, username: newUsername.trim() } : null
      );
      setIsEditingUsername(false);
    }
    setUsernameLoading(false);
  };

  const handleViewPhoto = () => {
    setShowPhotoOptions(false);
    setShowPhotoModal(true);
  };

  const handleChangePhoto = async () => {
    setShowPhotoOptions(false);
    let result = await ImagePicker.launchImageLibraryAsync({
      allowsEditing: true,
      aspect: [1, 1],
      quality: 0.8,
    });
    if (!result.canceled) {
      try {
        const fileUri = result.assets[0].uri;
        const fileExt = fileUri.split('.').pop();
        const fileName = `${profile?.id}/${Date.now()}.${fileExt}`;
        const base64 = await FileSystem.readAsStringAsync(fileUri, { encoding: FileSystem.EncodingType.Base64 });
        const fileBytes = Buffer.from(base64, 'base64');
        const { error: uploadError } = await supabase.storage
          .from('avatars')
          .upload(fileName, fileBytes, { contentType: 'image/jpeg', upsert: true });
        if (uploadError) {
          Alert.alert('Photo upload failed.', uploadError.message);
          return;
        }
        const { data: urlData } = supabase.storage.from('avatars').getPublicUrl(fileName);
        await supabase
          .from('profiles')
          .update({ photo_url: fileName, updated_at: new Date().toISOString() })
          .eq('id', profile?.id);
        setPhotoUrl(urlData?.publicUrl || null);
      } catch (error: any) {
        Alert.alert('Photo upload failed.', error.message);
      }
    }
  };

  if (loading) {
    return (
      <View className="flex-1 items-center justify-center bg-gray-100">
        <ActivityIndicator size="large" color="#FF6B00" />
      </View>
    );
  }

  if (!profile) {
    return (
      <View className="flex-1 items-center justify-center bg-gray-100">
        <Text className="text-gray-600">No profile data found.</Text>
      </View>
    );
  }

  return (
    <View 
      className="flex-1 bg-gray-100"
      style={{ paddingTop: insets.top }}
    >
      <AppHeaderProfile />

      <ScrollView
        {...panResponder.panHandlers}
        className="flex-1 p-4"
        contentContainerStyle={{ paddingBottom: 150 }}
        keyboardShouldPersistTaps="handled"
      >
        {/* --- Centered Profile Photo + Username Block --- */}
        <View style={{ alignItems: 'center', justifyContent: 'center', marginVertical: 38 }}>
          <View style={{ justifyContent: 'center', alignItems: 'center', position: 'relative', marginBottom: 10 }}>
            <Image
              source={{ uri: photoUrl || 'https://ui-avatars.com/api/?name=User&background=999&color=fff' }}
              style={{
                width: 200,
                height: 200,
                borderRadius: 100,
                backgroundColor: '#eee',
              }}
            />
            <TouchableOpacity
              style={{
                position: 'absolute',
                right: -10,
                bottom: 16,
                backgroundColor: '#fff',
                borderRadius: 18,
                elevation: 3,
                padding: 6,
              }}
              onPress={() => setShowPhotoOptions(true)}
            >
              <MaterialIcons name="edit" size={26} color="#FF6B00" />
            </TouchableOpacity>
          </View>

          {/* Username */}
          <View style={{ alignItems: 'center', width: 220, marginTop: 2 }}>
            {!isEditingUsername ? (
              <TouchableOpacity
                style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'center' }}
                onPress={() => setIsEditingUsername(true)}
                activeOpacity={0.7}
              >
                <Text style={{ fontSize: 24, fontWeight: 'bold', color: '#222', marginRight: 6 }}>
                  {profile.username}
                </Text>
                <MaterialIcons name="edit" size={22} color="#FF6B00" />
              </TouchableOpacity>
            ) : (
              <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'center' }}>
                <TextInput
                  style={{
                    fontSize: 22,
                    borderBottomWidth: 1,
                    borderColor: '#FF6B00',
                    color: '#333',
                    flex: 1,
                    textAlign: 'center',
                  }}
                  value={newUsername}
                  onChangeText={setNewUsername}
                  editable={!usernameLoading}
                  autoCapitalize="none"
                />
                <TouchableOpacity onPress={handleSaveUsername} disabled={usernameLoading} style={{ marginLeft: 10 }}>
                  <Text style={{ color: '#FF6B00', fontWeight: 'bold' }}>Save</Text>
                </TouchableOpacity>
                <TouchableOpacity onPress={() => setIsEditingUsername(false)} disabled={usernameLoading} style={{ marginLeft: 10 }}>
                  <Text style={{ color: '#888' }}>Cancel</Text>
                </TouchableOpacity>
              </View>
            )}
          </View>
        </View>

        {/* --- Fullscreen Photo Modal --- */}
        <Modal visible={showPhotoModal} transparent animationType="fade" onRequestClose={() => setShowPhotoModal(false)}>
          <Pressable
            style={{
              flex: 1,
              backgroundColor: 'rgba(20,20,20,0.96)',
              alignItems: 'center',
              justifyContent: 'center',
            }}
            onPress={() => setShowPhotoModal(false)}
          >
            <Image
              source={{ uri: photoUrl || undefined }}
              style={{
                width: '80%',
                height: '50%',
                borderRadius: 16,
                borderWidth: 4,
                borderColor: '#fff',
                resizeMode: 'cover',
              }}
            />
            <TouchableOpacity
              style={{
                position: 'absolute',
                top: 32,
                right: 20,
                padding: 8,
                backgroundColor: 'rgba(0,0,0,0.35)',
                borderRadius: 18,
              }}
              onPress={() => setShowPhotoModal(false)}
            >
              <AntDesign name="closecircle" size={32} color="#fff" />
            </TouchableOpacity>
          </Pressable>
        </Modal>

        {/* --- Bottom Sheet for Photo Options --- */}
        <Modal visible={showPhotoOptions} transparent animationType="slide">
          <Pressable
            style={{
              flex: 1,
              backgroundColor: 'rgba(0,0,0,0.4)',
              justifyContent: 'flex-end',
            }}
            onPress={() => setShowPhotoOptions(false)}
          >
            <View
              style={{
                backgroundColor: '#fff',
                borderTopLeftRadius: 20,
                borderTopRightRadius: 20,
                paddingVertical: 20,
                paddingHorizontal: 25,
              }}
            >
              <Text style={{ fontSize: 18, fontWeight: 'bold', color: '#333', marginBottom: 12 }}>
                Edit Profile Photo
              </Text>

              <TouchableOpacity
                onPress={handleViewPhoto}
                style={{
                  flexDirection: 'row',
                  alignItems: 'center',
                  paddingVertical: 14,
                }}
              >
                <MaterialIcons name="visibility" size={24} color="#FF6B00" style={{ marginRight: 12 }} />
                <Text style={{ fontSize: 16, color: '#222' }}>View Photo</Text>
              </TouchableOpacity>

              <TouchableOpacity
                onPress={handleChangePhoto}
                style={{
                  flexDirection: 'row',
                  alignItems: 'center',
                  paddingVertical: 14,
                }}
              >
                <MaterialIcons name="photo-camera" size={24} color="#FF6B00" style={{ marginRight: 12 }} />
                <Text style={{ fontSize: 16, color: '#222' }}>Change Photo</Text>
              </TouchableOpacity>

              <TouchableOpacity
                onPress={() => setShowPhotoOptions(false)}
                style={{
                  marginTop: 10,
                  alignItems: 'center',
                  paddingVertical: 12,
                  backgroundColor: '#f5f5f5',
                  borderRadius: 12,
                }}
              >
                <Text style={{ fontSize: 16, fontWeight: 'bold', color: '#FF6B00' }}>Cancel</Text>
              </TouchableOpacity>
            </View>
          </Pressable>
        </Modal>

        {/* Progress */}
        <Text className="text-xl font-bold text-gray-800 mb-4">Your Progress</Text>
        <View className="flex-row flex-wrap justify-between mb-8">
          {Object.entries(progressData).map(([key, value]) => (
            <View
              key={key}
              className="w-[48%] bg-white rounded-xl p-4 shadow-md mb-4 items-center"
            >
              <Text className="text-3xl font-bold text-[#FF6B00]">{value}</Text>
              <Text className="text-sm text-gray-500 text-center capitalize">
                {key.replace(/([A-Z])/g, ' $1')}
              </Text>
            </View>
          ))}
        </View>

        {/* Settings */}
        <Text className="text-lg font-bold text-gray-800 mb-4">Settings</Text>
        <View className="bg-white rounded-xl p-4 shadow-md mb-6">
          {(
            [
              ['Sound Effects', isSoundEffectsEnabled, setSoundEffectsEnabled],
              ['Vibration', isVibrationEnabled, setVibrationEnabled],
              ['Notifications', areNotificationsEnabled, setNotificationsEnabled],
              ['Dark Mode', isDarkModeEnabled, setDarkModeEnabled],
            ] as [string, boolean, React.Dispatch<React.SetStateAction<boolean>>][]
          ).map(([label, value, setter], idx) => (
            <View
              key={idx}
              className={`flex-row items-center justify-between py-2 ${idx !== 3 ? 'border-b border-gray-200' : ''
                }`}
            >
              <Text className="text-base font-medium text-gray-800">{label}</Text>
              <Switch
                trackColor={{ false: '#E5E7EB', true: '#FF6B00' }}
                thumbColor={value ? '#fff' : '#f4f3f4'}
                onValueChange={() => setter(!value)}
                value={value}
              />
            </View>
          ))}
        </View>

        {/* Account */}
        <Text className="text-lg font-bold text-gray-800 mb-4">Account</Text>
        <View className="bg-white rounded-xl p-4 shadow-md mb-6">
          <TouchableOpacity
            onPress={handleSignOut}
            className="flex-row items-center justify-between py-2"
          >
            <Text className="text-base font-bold text-red-500">Sign Out</Text>
            <AntDesign name="logout" size={24} color="#ef4444" />
          </TouchableOpacity>
        </View>
      </ScrollView>
    </View>
  );
};

export default Profile;

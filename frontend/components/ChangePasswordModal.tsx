// File: frontend/components/ChangePasswordModal.tsx
import React, { useState } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  Modal,
  ActivityIndicator,
  Alert,
  KeyboardAvoidingView,
  Platform,
  TouchableWithoutFeedback,
  Keyboard,
} from 'react-native';
import { MaterialIcons, Ionicons } from '@expo/vector-icons';
import { supabase } from '../src/supabaseClient';
import { useTheme } from '../src/ThemeContext';


interface ChangePasswordModalProps {
  visible: boolean;
  onClose: () => void;
}


const ChangePasswordModal: React.FC<ChangePasswordModalProps> = ({ visible, onClose }) => {
  const { isDark } = useTheme();
  
  const [oldPassword, setOldPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [loading, setLoading] = useState(false);
  
  // Visibility toggles
  const [showOldPassword, setShowOldPassword] = useState(false);
  const [showNewPassword, setShowNewPassword] = useState(false);


  // Colors based on your theme
  const bgColor = isDark ? 'bg-darksurface' : 'bg-white';
  const textColor = isDark ? 'text-secondary' : 'text-primary';
  const inputBg = isDark ? 'bg-darkbg' : 'bg-secondary';
  const borderColor = isDark ? 'border-highlight' : 'border-highlight';
  const placeholderColor = isDark ? '#A8A8A8' : '#888';


  const handleUpdatePassword = async () => {
    if (!oldPassword || !newPassword || !confirmPassword) {
      Alert.alert('Error', 'Please fill in all fields.');
      return;
    }


    if (newPassword.length < 6) {
      Alert.alert('Weak Password', 'New password must be at least 6 characters long.');
      return;
    }


    if (newPassword !== confirmPassword) {
      Alert.alert('Mismatch', 'New passwords do not match.');
      return;
    }


    if (oldPassword === newPassword) {
      Alert.alert('Error', 'New password cannot be the same as the old password.');
      return;
    }


    setLoading(true);


    try {
      // 1. Get current user email
      const { data: { user }, error: userError } = await supabase.auth.getUser();
      
      if (userError || !user || !user.email) {
        throw new Error('Unable to identify user. Please sign in again.');
      }


      // 2. Verify Old Password by attempting to sign in
      const { error: signInError } = await supabase.auth.signInWithPassword({
        email: user.email,
        password: oldPassword,
      });


      if (signInError) {
        Alert.alert('Incorrect Password', 'The old password you entered is incorrect.');
        setLoading(false);
        return;
      }


      // 3. Update to New Password
      const { error: updateError } = await supabase.auth.updateUser({
        password: newPassword,
      });


      if (updateError) throw updateError;


      Alert.alert('Success', 'Your password has been updated successfully.', [
        { 
          text: 'OK', 
          onPress: () => {
            resetForm();
            onClose();
          }
        }
      ]);
    } catch (error: any) {
      Alert.alert('Update Failed', error.message || 'An unexpected error occurred.');
    } finally {
      setLoading(false);
    }
  };


  const resetForm = () => {
    setOldPassword('');
    setNewPassword('');
    setConfirmPassword('');
    setShowOldPassword(false);
    setShowNewPassword(false);
  };


  const handleClose = () => {
    resetForm();
    onClose();
  };


  return (
    <Modal
      animationType="fade"
      transparent={true}
      visible={visible}
      onRequestClose={handleClose}
    >
      <TouchableWithoutFeedback onPress={Keyboard.dismiss}>
        <View className="flex-1 justify-center items-center bg-black/60 px-6">
          <KeyboardAvoidingView 
            behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
            className="w-full"
          >
            <View className={`w-full p-6 rounded-3xl border border-accent shadow-xl ${bgColor}`}>
              
              {/* Header */}
              <View className="items-center mb-6">
                <View className="w-12 h-12 rounded-full bg-accent/10 items-center justify-center mb-3">
                  <MaterialIcons name="lock-reset" size={28} color="#FF6B00" />
                </View>
                <Text className={`text-2xl font-audiowide text-center ${textColor}`}>
                  Change Password
                </Text>
                <Text className="text-sm font-montserrat-regular text-gray-500 text-center mt-1">
                  Verify your old password to set a new one.
                </Text>
              </View>


              {/* Old Password - Separated from new password fields */}
              <View className="mb-5">
                <View className={`flex-row items-center border rounded-xl px-4 py-3 ${inputBg} ${borderColor}`}>
                  <MaterialIcons name="lock" size={20} color="#FF6B00" style={{ marginRight: 10 }} />
                  <TextInput
                    placeholder="Old Password"
                    placeholderTextColor={placeholderColor}
                    secureTextEntry={!showOldPassword}
                    value={oldPassword}
                    onChangeText={setOldPassword}
                    className={`flex-1 text-base font-montserrat-semibold ${textColor}`}
                  />
                  <TouchableOpacity onPress={() => setShowOldPassword(!showOldPassword)}>
                    <Ionicons 
                      name={showOldPassword ? "eye-off-outline" : "eye-outline"} 
                      size={20} 
                      color={isDark ? '#A8A8A8' : '#666'} 
                    />
                  </TouchableOpacity>
                </View>
              </View>


              {/* Divider for visual separation */}
              <View className="flex-row items-center mb-4">
                <View className="flex-1 h-px bg-gray-300 dark:bg-gray-600" />
                <Text className="mx-3 text-xs font-montserrat-medium text-gray-400 dark:text-gray-500">
                  NEW PASSWORD
                </Text>
                <View className="flex-1 h-px bg-gray-300 dark:bg-gray-600" />
              </View>


              {/* New Password Fields */}
              <View className="space-y-3">
                {/* New Password */}
                <View className={`flex-row items-center border rounded-xl mb-4 px-4 py-3 ${inputBg} ${borderColor}`}>
                  <MaterialIcons name="lock-outline" size={20} color="#FF6B00" style={{ marginRight: 10 }} />
                  <TextInput
                    placeholder="New Password"
                    placeholderTextColor={placeholderColor}
                    secureTextEntry={!showNewPassword}
                    value={newPassword}
                    onChangeText={setNewPassword}
                    className={`flex-1 text-base font-montserrat-semibold ${textColor}`}
                  />
                </View>


                {/* Confirm Password */}
                <View className={`flex-row items-center border rounded-xl px-4 py-3 gap-y-3 ${inputBg} ${borderColor}`}>
                  <MaterialIcons name="lock-outline" size={20} color="#FF6B00" style={{ marginRight: 10 }} />
                  <TextInput
                    placeholder="Confirm New Password"
                    placeholderTextColor={placeholderColor}
                    secureTextEntry={!showNewPassword}
                    value={confirmPassword}
                    onChangeText={setConfirmPassword}
                    className={`flex-1 text-base font-montserrat-semibold ${textColor}`}
                  />
                  <TouchableOpacity onPress={() => setShowNewPassword(!showNewPassword)}>
                    <Ionicons 
                      name={showNewPassword ? "eye-off-outline" : "eye-outline"} 
                      size={20} 
                      color={isDark ? '#A8A8A8' : '#666'} 
                    />
                  </TouchableOpacity>
                </View>
              </View>


              {/* Actions */}
              <View className="flex-row justify-between mt-8">
                <TouchableOpacity
                  onPress={handleClose}
                  disabled={loading}
                  className={`flex-1 py-3 rounded-full border mr-2 items-center justify-center ${
                    isDark ? 'border-neutral' : 'border-gray-300'
                  }`}
                >
                  <Text className={`font-fredoka-bold ${isDark ? 'text-neutral' : 'text-gray-500'}`}>
                    Cancel
                  </Text>
                </TouchableOpacity>


                <TouchableOpacity
                  onPress={handleUpdatePassword}
                  disabled={loading}
                  className="flex-1 bg-accent py-3 rounded-full ml-2 items-center justify-center shadow-md"
                >
                  {loading ? (
                    <ActivityIndicator size="small" color="#fff" />
                  ) : (
                    <Text className="text-white font-fredoka-bold">Update</Text>
                  )}
                </TouchableOpacity>
              </View>


            </View>
          </KeyboardAvoidingView>
        </View>
      </TouchableWithoutFeedback>
    </Modal>
  );
};


export default ChangePasswordModal;

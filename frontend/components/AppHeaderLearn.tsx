import React, { useState } from 'react';
import { View, Text, TouchableOpacity, Modal } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons, Feather } from '@expo/vector-icons';
import { useRouter } from 'expo-router';

interface AppHeaderLearnProps {
  title: string;
  completedCount: number;
  totalCount: number;
  onResetProgress?: () => void;
}

const AppHeaderLearn: React.FC<AppHeaderLearnProps> = ({
  title,
  completedCount,
  totalCount,
  onResetProgress
}) => {
  const router = useRouter();
  const pct = totalCount > 0 ? Math.min(100, Math.round((completedCount / totalCount) * 100)) : 0;
  const [showConfirm, setShowConfirm] = useState(false);

  return (
    <View style={{ backgroundColor: 'transparent' }}>
      <LinearGradient
        colors={['#FF6B00', '#FFAB7B']}
        className="py-3 px-4 flex-row items-center justify-between"
      >
        {/* Back button */}
        <TouchableOpacity onPress={() => router.back()} className="p-1">
          <Ionicons name="arrow-back" size={24} color="black" />
        </TouchableOpacity>

        {/* Title and progress */}
        <View className="flex-1 items-center -ml-8">
          <Text className="text-primary text-lg font-fredoka-semibold">{title}</Text>
          <Text className="text-primary text-xs mt-1 font-fredoka">
            {completedCount}/{totalCount} completed
          </Text>
        </View>

        {/* Reset icon at upper right */}
        <TouchableOpacity
          onPress={() => setShowConfirm(true)}
          className="p-1"
          style={{ width: 32, alignItems: 'flex-end' }}
        >
          <Feather name="rotate-ccw" size={22} color="#472900" />
        </TouchableOpacity>
      </LinearGradient>

      {/* Progress bar */}
      <View className="w-full bg-secondary rounded-full" style={{ height: 5 }}>
        <View
          style={{ width: `${pct}%`, height: 5 }}
          className="bg-[#FF6B00] rounded-full"
        />
      </View>

      {/* Bottom fade gradient */}
      <LinearGradient
        colors={[
          'rgba(255, 171, 123, 1.0)',
          'rgba(255, 171, 123, 0.0)',
        ]}
        start={{ x: 0.5, y: 0.0 }}
        end={{ x: 0.5, y: 1.0 }}
        className="items-center py-2"
      />

      {/* Minimalist modal confirmation */}
      <Modal
        visible={showConfirm}
        transparent
        animationType="fade"
        onRequestClose={() => setShowConfirm(false)}
      >
        <View style={{
          flex: 1,
          backgroundColor: 'rgba(0,0,0,0.25)',
          justifyContent: 'center',
          alignItems: 'center'
        }}>
          <View style={{
            backgroundColor: '#FFF3E6',
            padding: 19,
            borderRadius: 14,
            width: 250,
            alignItems: 'center',
            shadowColor: '#FF6B00',
            shadowOpacity: 0.12,
            shadowRadius: 8,
            elevation: 5
          }}>
            <Text style={{
              fontFamily: 'Fredoka-SemiBold',
              fontSize: 16,
              color: '#472900',
              marginBottom: 9
            }}>Reset lesson progress?</Text>
            <Text style={{
              fontFamily: 'Montserrat-SemiBold',
              fontSize: 13,
              color: '#472900',
              marginBottom: 17,
              textAlign: 'center'
            }}>
              This will restart completion. Are you sure?
            </Text>
            <View style={{ flexDirection: 'row', marginTop: 2 }}>
              <TouchableOpacity
                style={{
                  backgroundColor: '#FF6B00',
                  paddingVertical: 8,
                  paddingHorizontal: 18,
                  borderRadius: 8,
                  marginRight: 12
                }}
                onPress={() => {
                  setShowConfirm(false);
                  if (onResetProgress) onResetProgress();
                }}
              >
                <Text style={{
                  color: 'white',
                  fontSize: 14,
                  fontFamily: 'Fredoka-SemiBold'
                }}>Reset</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={{
                  backgroundColor: '#f2ebe8',
                  paddingVertical: 8,
                  paddingHorizontal: 18,
                  borderRadius: 8
                }}
                onPress={() => setShowConfirm(false)}
              >
                <Text style={{
                  color: '#472900',
                  fontSize: 14,
                  fontFamily: 'Fredoka-SemiBold'
                }}>Cancel</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
    </View>
  );
};

export default AppHeaderLearn;
